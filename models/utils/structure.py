import torch
import spconv.pytorch as spconv

try:
    import ocnn
except ImportError:
    ocnn = None
from addict import Dict

from models.utils.serialization import encode, decode
from models.utils import offset2batch, batch2offset



class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)
        
        if "bt" not in self.keys() and "batch" in self.keys():
            tn_sq = self.tn.squeeze()
            if tn_sq.dim() == 0:
                tn_sq = tn_sq.unsqueeze(0)
            self["bt"] = len(torch.bincount(tn_sq.int())) * self.batch + self.tn.squeeze()
            
    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(self.coord.size(0), device=order.device).repeat(
                order.shape[0], 1
            ),
        )
        # order鎸囨帓搴忓悗鐨勫簭鍒楀搴斿師濮嬬殑绱㈠紩
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse
        # print("======DENUG==========")
        # print(f"self.offset: {self.offset}")
        # print(f"self.batch.shape: {self.batch.shape}")
        
        # print(f"order.shape: {order.shape}")
        
        # print(f"self.grid_coord.shape: {self.grid_coord.shape}")
        # print(f"self.grid_coord: {self.grid_coord}")
        # print(f"self.coord.shape: {self.coord.shape}")
        # print(f"self.coord: {self.coord}")
        
    def time_serialization(self):
        order_ = torch.argsort(self.coord[:, 2]+2*self.batch, descending=False)
        order_descend = torch.argsort(self.coord[:, 2]+2*self.batch, descending=True)
        order = torch.cat([order_.unsqueeze(0), order_descend.unsqueeze(0)], dim=0)  
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(self.coord.size(0), device=order.device).repeat(
                order.shape[0], 1
            ),
        )
        self["t_serialized_order"] = order
        self["t_serialized_inverse"] = inverse
        # print("======DENUG==========")
        # print(f"order.shape: {order.shape}")
        # print(f"inverse.shape: {inverse.shape}")
        # print(self["serialized_code"].shape)
        # print(self.coord)
        # print(self.coord[order[0]])
        # print(self.coord[order[0]][inverse[0]])
        # raise
        
    def sparsify(self, pad=96, wtn=True):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat if wtn else self.feat[:, :-1],
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn."
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
        
        
# without time_window
class Point_V2(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)
            
    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(self.coord.size(0), device=order.device).repeat(
                order.shape[0], 1
            ),
        )
        # order鎸囨帓搴忓悗鐨勫簭鍒楀搴斿師濮嬬殑绱㈠紩
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse
        # print("======DENUG==========")
        # print(f"self.offset: {self.offset}")
        # print(f"self.batch.shape: {self.batch.shape}")
        
        # print(f"order.shape: {order.shape}")
        
        # print(f"self.grid_coord.shape: {self.grid_coord.shape}")
        # print(f"self.grid_coord: {self.grid_coord}")
        # print(f"self.coord.shape: {self.coord.shape}")
        # print(f"self.coord: {self.coord}")
        
    @torch.no_grad()
    def time_serialization(self):
        # Keep samples in ascending batch order for BOTH scan directions.
        # Only reverse event order WITHIN each sample.
        forward_orders, backward_orders = [], []
        for batch_id in self.batch.unique(sorted=True):
            idx = torch.nonzero(self.batch == batch_id, as_tuple=False).reshape(-1)
            local_t = self.coord[idx, 2]
            forward_orders.append(idx[torch.argsort(local_t, descending=False)])
            backward_orders.append(idx[torch.argsort(local_t, descending=True)])
        order = torch.stack([torch.cat(forward_orders), torch.cat(backward_orders)], dim=0)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order,
            src=torch.arange(self.coord.size(0), device=order.device).repeat(2, 1),
        )
        self["t_serialized_order"] = order
        self["t_serialized_inverse"] = inverse
        # print("======DENUG==========")
        # print(f"order.shape: {order.shape}")
        # print(f"inverse.shape: {inverse.shape}")
        # print(self["serialized_code"].shape)
        # print(self.coord)
        # print(self.coord[order[0]])
        # print(self.coord[order[0]][inverse[0]])
        # raise
        
    def sparsify(self, pad=96, wtn=True):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def octreetization(self, depth=None, full_depth=None):
        """
        Point Cloud Octreelization

        Generate octree with OCNN
        relay on ["grid_coord", "batch", "feat"]
        """
        assert (
            ocnn is not None
        ), "Please follow https://github.com/octree-nn/ocnn-pytorch install ocnn."
        assert {"grid_coord", "feat", "batch"}.issubset(self.keys())
        # add 1 to make grid space support shift order
        if depth is None:
            if "depth" in self.keys():
                depth = self.depth
            else:
                depth = int(self.grid_coord.max() + 1).bit_length()
        if full_depth is None:
            full_depth = 2
        self["depth"] = depth
        assert depth <= 16  # maximum in ocnn

        # [0, 2**depth] -> [0, 2] -> [-1, 1]
        coord = self.grid_coord / 2 ** (self.depth - 1) - 1.0
        point = ocnn.octree.Points(
            points=coord,
            features=self.feat,
            batch_id=self.batch.unsqueeze(-1),
            batch_size=self.batch[-1] + 1,
        )
        octree = ocnn.octree.Octree(
            depth=depth,
            full_depth=full_depth,
            batch_size=self.batch[-1] + 1,
            device=coord.device,
        )
        octree.build_octree(point)
        octree.construct_all_neigh()
        self["octree"] = octree
        
        
        
        
        
        
        
