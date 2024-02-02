import meshio
import numpy as np
from typing import Union

from ..ipc_utils.logging_utils import ipc_logger


def np_to_tuple(np_array):
    return tuple(map(tuple, np_array))


class IPCTriMesh:
    def __init__(
        self,
        filename: str = None,
        scale: Union[float, list, np.ndarray] = 1.0,
        vertices: np.ndarray = None,
        triangles: np.ndarray = None,
    ):
        """Initialize triangular mesh either from file or from vertices and triangles.

        Args:
            filename (str, optional): Path to a triangular mesh file supported by meshio (e.g. .obj file). Defaults to None.
            scale (Union[float, list, np.ndarray], optional): Scaling coefficient of the initial mesh. Defaults to 1.0.
            vertices (np.ndarray, optional): Initial mesh vertices; shape=(N, 3), dtype=float32. Defaults to None.
            triangles (np.ndarray, optional): Initial mesh triangles; shape=(N, 3), dtype=int32. Defaults to None.
        """
        self.vertices = np.zeros((0, 3), dtype=np.float32)
        self.triangles = np.zeros((0, 3), dtype=np.int32)
        self.edges = set()

        scale = np.array(scale, dtype=np.float32)
        assert scale.ndim == 0 or (scale.ndim == 1 and len(scale) == 3)

        if filename is not None and vertices is None and triangles is None:
            mesh = meshio.read(filename)

            self.vertices = mesh.points.astype(np.float32) * scale
            if "triangle" in mesh.cells_dict:
                self.triangles = mesh.cells_dict["triangle"]
            if "line" in mesh.cells_dict:
                self.edges.update(np_to_tuple(np.sort(mesh.cells_dict["line"], axis=1)))
        elif filename is None and vertices is not None and triangles is not None:
            self.vertices = vertices.astype(np.float32) * scale
            self.triangles = triangles.astype(np.int32)

            assert vertices.ndim == 2 and vertices.shape[1] == 3
            assert triangles.ndim == 2 and triangles.shape[1] == 3
        elif filename is None and (vertices is None or triangles is None):
            raise ValueError(
                "Either filename or vertices and triangles must be provided."
            )
        else:
            raise ValueError(
                "filename and vertices/triangles cannot be provided together."
            )
        
        assert np.all(np.isfinite(self.vertices))

        ######## Get edges ########

        for i, j, k in self.triangles:
            self.edges.add(tuple(sorted((i, j))))
            self.edges.add(tuple(sorted((j, k))))
            self.edges.add(tuple(sorted((k, i))))

        if len(self.edges) == 0:
            self.edges = np.zeros((0, 2), dtype=np.int32)
        else:
            self.edges = np.array(list(self.edges), dtype=np.int32)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    @property
    def surface_vertices(self) -> np.ndarray:
        return np.arange(self.n_vertices, dtype=np.int32)

    @property
    def surface_edges(self) -> np.ndarray:
        return self.edges

    @property
    def surface_triangles(self) -> np.ndarray:
        return self.triangles

    @property
    def n_surface_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_surface_edges(self) -> int:
        return len(self.edges)

    @property
    def n_surface_triangles(self) -> int:
        return len(self.triangles)


class IPCTetMesh:
    def __init__(
        self,
        filename: str = None,
        scale: Union[float, list, np.ndarray] = 1.0,
        vertices: np.ndarray = None,
        tets: np.ndarray = None,
    ):
        """Initialize tetrahedral mesh either from file or from vertices and tets.

        Args:
            filename (str, optional): Path to a tetrahedral mesh file supported by meshio (e.g. Gmsh .msh file). Defaults to None.
            scale (Union[float, list, np.ndarray], optional): Scaling coefficient of the initial mesh. Defaults to 1.0.
            vertices (np.ndarray, optional): Vertices of the initial mesh; shape=(N, 3), dtype=float32. Defaults to None.
            tets (np.ndarray, optional): Tetrahedra of the initial mesh; shape=(N, 4), dtype=int32. Defaults to None.
        """

        self.vertices = np.zeros((0, 3), dtype=np.float32)
        self.tets = np.zeros((0, 4), dtype=np.int32)
        # self.surface_vertices = np.zeros((0,), dtype=np.int32)
        # self.surface_edges = np.zeros((0, 2), dtype=np.int32)
        # self.surface_triangles = np.zeros((0, 3), dtype=np.int32)
        self.surface_vertices = set()
        self.surface_edges = set()
        self.surface_triangles = set()

        scale = np.array(scale, dtype=np.float32)
        assert scale.ndim == 0 or (scale.ndim == 1 and len(scale) == 3)

        if filename is not None and vertices is None and tets is None:
            # Initialize from file
            mesh = meshio.read(filename)

            self.vertices = mesh.points.astype(np.float32) * scale
            if "tetra" in mesh.cells_dict:
                self.tets = mesh.cells_dict["tetra"]
            if "triangle" in mesh.cells_dict:
                self.surface_triangles.update(np_to_tuple(mesh.cells_dict["triangle"]))
            if "line" in mesh.cells_dict:
                self.surface_edges.update(
                    np_to_tuple(np.sort(mesh.cells_dict["line"], axis=1))
                )
                self.surface_vertices.update(
                    tuple(np.unique(mesh.cells_dict["line"].reshape(-1)))
                )
            if "vertex" in mesh.cells_dict:
                self.surface_vertices.update(
                    tuple(mesh.cells_dict["vertex"].reshape(-1))
                )
        elif filename is None and vertices is not None and tets is not None:
            # Initialize from vertices and tets
            self.vertices = vertices.astype(np.float32) * scale
            self.tets = tets.astype(np.int32)

            assert vertices.ndim == 2 and vertices.shape[1] == 3
            assert tets.ndim == 2 and tets.shape[1] == 4
        elif filename is None and (vertices is None or tets is None):
            raise ValueError("Either filename or vertices and tets must be provided.")
        else:
            raise ValueError("filename and vertices/tets cannot be provided together.")

        ######## Get surface triangles ########

        faces = {}  # dict of open faces (key: tuple of sorted indices)

        def add_face(i, j, k):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        for v0, v1, v2, v3 in self.tets:
            add_face(v0, v2, v1)
            add_face(v1, v2, v3)
            add_face(v0, v1, v3)
            add_face(v0, v3, v2)

        self.surface_triangles.update(faces.values())

        ######## Get surface edges and vertices from surface triangles ########

        for i, j, k in self.surface_triangles:
            self.surface_vertices.add(i)
            self.surface_vertices.add(j)
            self.surface_vertices.add(k)

        for i, j, k in self.surface_triangles:
            self.surface_edges.add(tuple(sorted((i, j))))
            self.surface_edges.add(tuple(sorted((j, k))))
            self.surface_edges.add(tuple(sorted((k, i))))

        if len(self.surface_triangles) == 0:
            self.surface_triangles = np.zeros((0, 3), dtype=np.int32)
        else:
            self.surface_triangles = np.array(
                list(self.surface_triangles), dtype=np.int32
            )
        if len(self.surface_edges) == 0:
            self.surface_edges = np.zeros((0, 2), dtype=np.int32)
        else:
            self.surface_edges = np.array(list(self.surface_edges), dtype=np.int32)
        if len(self.surface_vertices) == 0:
            self.surface_vertices = np.zeros((0,), dtype=np.int32)
        else:
            self.surface_vertices = np.array(
                list(self.surface_vertices), dtype=np.int32
            )

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_tets(self) -> int:
        return len(self.tets)

    @property
    def n_surface_vertices(self) -> int:
        return len(self.surface_vertices)

    @property
    def n_surface_edges(self) -> int:
        return len(self.surface_edges)

    @property
    def n_surface_triangles(self) -> int:
        return len(self.surface_triangles)
