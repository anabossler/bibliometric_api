"""gr.File() component"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import gradio_client.utils as client_utils
from gradio_client import handle_file

import gradio as gr
from gradio import processing_utils
from gradio.components.base import Component
from gradio.data_classes import FileData, ListFiles
from gradio.events import Events
from gradio.utils import NamedString
import ase.io
import numpy as np

if TYPE_CHECKING:
    from gradio.components import Timer


def write_proteindatabank(fileobj, images, write_arrays=True):
    """Write images to PDB-file.
    From ASE, but edited to not use the upper-case element symbols.
    """
    rot_t = None
    if hasattr(images, 'get_positions'):
        images = [images]

    #     1234567 123 6789012345678901   89   67   456789012345678901234567 890
    format = ('ATOM  %5d %4s %4s %4d    %8.3f%8.3f%8.3f%6.2f%6.2f'
              '          %2s  \n')

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)

    for n, atoms in enumerate(images):
        if atoms.get_pbc().any():
            currentcell = atoms.get_cell()
            cellpar = currentcell.cellpar()
            _, rot_t = currentcell.standard_form()
            # ignoring Z-value, using P1 since we have all atoms defined
            # explicitly
            cellformat = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
            fileobj.write(cellformat % (cellpar[0], cellpar[1], cellpar[2],
                                        cellpar[3], cellpar[4], cellpar[5]))
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        p = atoms.get_positions()
        if rot_t is not None:
            p = p.dot(rot_t.T)
        occupancy = np.ones(len(atoms))
        bfactor = np.zeros(len(atoms))
        residuenames = ['MOL '] * len(atoms)
        residuenumbers = np.ones(len(atoms))
        names = atoms.get_chemical_symbols()
        if write_arrays:
            if 'occupancy' in atoms.arrays:
                occupancy = atoms.get_array('occupancy')
            if 'bfactor' in atoms.arrays:
                bfactor = atoms.get_array('bfactor')
            if 'residuenames' in atoms.arrays:
                residuenames = atoms.get_array('residuenames')
            if 'residuenumbers' in atoms.arrays:
                residuenumbers = atoms.get_array('residuenumbers')
            if 'atomtypes' in atoms.arrays:
                names = atoms.get_array('atomtypes')
        for a in range(natoms):
            x, y, z = p[a]
            occ = occupancy[a]
            bf = bfactor[a]
            resname = residuenames[a].ljust(4)
            resseq = residuenumbers[a]
            name = names[a]
            fileobj.write(format % ((a + 1) % MAXNUM, name, resname, resseq,
                                    x, y, z, occ, bf, symbols[a]))
        fileobj.write('ENDMDL\n')

def find_minimum_repeats(atoms, min_length):
    """
    Find the minimum number of repeats in each unit cell direction
    to meet at least `min_length` angstroms.
    Parameters:
        atoms (ase.Atoms): The ASE Atoms object.
        min_length (float): The minimum length required in each direction.
    Returns:
        tuple: A tuple of integers representing the number of repeats
               in the x, y, and z directions.
    """
    cell_lengths = atoms.get_cell().lengths()  # Get the lengths of the unit cell
    repeats = [max(1, int(np.ceil(min_length / length))) for length in cell_lengths]
    return tuple(repeats)

def convert_file_to_pdb(file_path: str | Path, gradio_cache: str | Path) -> str:
    # Read the file using ASE, and convert even if it's pdb to make sure all elements go lower case

    try:
        structures = ase.io.read(file_path, ':')
    except Exception as e:
        # Bad upload structure, no need to visualize
        raise gr.Error(f'Error parsing file with ase: {str(e)}')

    if all(structures[0].pbc):
        # find the minimum number of repeats in each unit cell direction to meet at least 20 angstroms
        repeats = find_minimum_repeats(structures[0], min_length=15.0)

    structures = [s.repeat(repeats) if all(s.pbc) else s for s in structures]

    # Create a temporary PDB file
    with tempfile.NamedTemporaryFile(
        delete=False, dir=gradio_cache, suffix=".pdb", mode='w',
    ) as temp_pdb_file:
        write_proteindatabank(temp_pdb_file, structures)
    file_name = temp_pdb_file.name
    return file_name
   

class Molecule3D(Component):
    """
    Creates a file component that allows uploading one or more generic files (when used as an input) or displaying generic files or URLs for download (as output).
    Demo: zip_files, zip_to_json
    """

    EVENTS = [Events.change, Events.select, Events.clear, Events.upload, Events.delete]

    def __init__(
        self,
        value: str | list[str] | Callable | None = None,
        reps: Any | None = [],
        config: Any | None = {
    "backgroundColor": "white",
    "orthographic": False,
    "disableFog": False,
  },
        confidenceLabel: str | None = "pLDDT",
        *,
        file_count: Literal["single", "multiple", "directory"] = "single",
        file_types: list[str] | None = None,
        type: Literal["filepath", "binary"] = "filepath",
        label: str | None = None,
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        container: bool = True,
        scale: int | None = None,
        min_width: int = 160,
        height: int | float | None = None,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,
        showviewer: bool = True      
    ):
        """
        Parameters:
            value: Default file(s) to display, given as a str file path or URL, or a list of str file paths / URLs. If callable, the function will be called whenever the app loads to set the initial value of the component.
            file_count: if single, allows user to upload one file. If "multiple", user uploads multiple files. If "directory", user uploads all files in selected directory. Return type will be list for each file in case of "multiple" or "directory".
            file_types: List of file extensions or types of files to be uploaded (e.g. ['image', '.json', '.mp4']). "file" allows any file to be uploaded, "image" allows only image files to be uploaded, "audio" allows only audio files to be uploaded, "video" allows only video files to be uploaded, "text" allows only text files to be uploaded.
            representations: list of representation objects
            config: dictionary of config options
            confidenceLabel: label for confidence values stored in the bfactor column of a pdb file
            type: Type of value to be returned by component. "file" returns a temporary file object with the same base name as the uploaded file, whose full path can be retrieved by file_obj.name, "binary" returns an bytes object.
            label: The label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.
            every: Continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            inputs: Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.
            show_label: if True, will display label.
            container: If True, will place the component in a container - providing some extra padding around the border.
            scale: relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            height: The maximum height of the file component, specified in pixels if a number is passed, or in CSS units if a string is passed. If more files are uploaded than can fit in the height, a scrollbar will appear.
            interactive: if True, will allow users to upload a file; if False, can only be used to display files. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
            key: if assigned, will be used to assume identity across a re-render. Components that have the same key across a re-render will have their value preserved.
            showviewer: If True, will display the 3Dmol.js viewer. If False, will not display the 3Dmol.js viewer.
        """
        file_count_valid_types = ["single", "multiple", "directory"]
        self.file_count = file_count

        if self.file_count not in file_count_valid_types:
            raise ValueError(
                f"Parameter file_count must be one of them: {file_count_valid_types}"
            )
        elif self.file_count in ["multiple", "directory"]:
            self.data_model = ListFiles
        else:
            self.data_model = FileData
        self.file_types = file_types
        if file_types is not None and not isinstance(file_types, list):
            raise ValueError(
                f"Parameter file_types must be a list. Received {file_types.__class__.__name__}"
            )
        valid_types = [
            "filepath",
            "binary",
        ]
        if type not in valid_types:
            raise ValueError(
                f"Invalid value for parameter `type`: {type}. Please choose from one of: {valid_types}"
            )
        if file_count == "directory" and file_types is not None:
            warnings.warn(
                "The `file_types` parameter is ignored when `file_count` is 'directory'."
            )
        super().__init__(
            label=label,
            every=every,
            inputs=inputs,
            show_label=show_label,
            container=container,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            render=render,
            key=key,
            value=value,
        )
        self.type = type
        self.height = height
        self.reps = reps
        self.config = config
        self.confidenceLabel = confidenceLabel
        self.showviewer = showviewer

    def _process_single_file(self, f: FileData) -> NamedString | bytes:
        file_name = f.path
        
        file_name = convert_file_to_pdb(file_name, self.GRADIO_CACHE)

        if self.type == "filepath":
            return NamedString(file_name)
        elif self.type == "binary":
            with open(file_name, "rb") as file_data:
                return file_data.read()
        else:
            raise ValueError(
                "Unknown type: "
                + str(type)
                + ". Please choose from: 'filepath', 'binary'."
            )

    def preprocess(
        self, payload: ListFiles | FileData | None
    ) -> bytes | str | list[bytes] | list[str] | None:
        """
        Parameters:
            payload: molecule3d information as a FileData object, or a list of FileData objects.
        Returns:
            Passes the file as a `str` or `bytes` object, or a list of `str` or list of `bytes` objects, depending on `type` and `file_count`.
        """
        if payload is None:
            return None

        if self.file_count == "single":
            if isinstance(payload, ListFiles):
                return self._process_single_file(payload[0])
            return self._process_single_file(payload)
        if isinstance(payload, ListFiles):
            return [self._process_single_file(f) for f in payload]  # type: ignore
        return [self._process_single_file(payload)]  # type: ignore

    def _download_files(self, value: str | list[str]) -> str | list[str]:
        downloaded_files = []
        if isinstance(value, list):
            for file in value:
                if client_utils.is_http_url_like(file):
                    downloaded_file = processing_utils.save_url_to_cache(
                        file, self.GRADIO_CACHE
                    )
                    downloaded_files.append(downloaded_file)
                else:
                    downloaded_files.append(file)
            return downloaded_files
        if client_utils.is_http_url_like(value):
            downloaded_file = processing_utils.save_url_to_cache(
                value, self.GRADIO_CACHE
            )
            return downloaded_file
        else:
            return value

    def postprocess(self, value: str | list[str] | None) -> ListFiles | FileData | None:
        """
        Parameters:
            value: Expects a `str` filepath or URL, or a `list[str]` of filepaths/URLs.
        Returns:
            molecule3d information as a FileData object, or a list of FileData objects.
        """
        if value is None:
            return None
        value = self._download_files(value)

        
        if isinstance(value, list):
            value = [convert_file_to_pdb(str(path), self.GRADIO_CACHE) for path in value]
            return ListFiles(
                root=[
                    FileData(
                        path=file,
                        orig_name=Path(file).name,
                        size=Path(file).stat().st_size,
                    )
                    for file in value
                ]
            )
        else:
            value = convert_file_to_pdb(str(value), self.GRADIO_CACHE)
            if value is not None:
                return FileData(
                    path=value,
                    orig_name=Path(value).name,
                    size=Path(value).stat().st_size,
                )
            else:
                return None

    def process_example(self, value: str | list | None) -> str:
        if value is None:
            return ""
        elif isinstance(value, list):
            return ", ".join([Path(file).name for file in value])
        else:
            return Path(value).name

    def example_payload(self) -> Any:
        if self.file_count == "single":
            return handle_file(
                "https://files.rcsb.org/view/1PGA.pdb"
            )
        else:
            return [
                handle_file(
                    "https://files.rcsb.org/view/1PGA.pdb"
                )
            ]

    def example_value(self) -> Any:
        if self.file_count == "single":
            return "https://files.rcsb.org/view/1PGA.pdb"
        else:
            return [
                "https://files.rcsb.org/view/1PGA.pdb"
            ]
