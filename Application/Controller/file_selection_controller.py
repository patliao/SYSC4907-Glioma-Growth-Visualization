from Application.Model.biological_model import BiologicalModel
from Application.View.file_selection_view import FileSelectionView


class FileSelectionController:
    _instance = None

    def __init__(self):
        self.view = FileSelectionView(self)
        self.equation_model = BiologicalModel.instance()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = FileSelectionController()
        return cls._instance

    def set_selected_file(self, file_key, file_path):
        self.equation_model.update_file_paths(file_key, file_path)
