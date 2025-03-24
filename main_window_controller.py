import sys
import os
import numpy as np

from UIUsedAIPrediction import UIUsedAIPrediction
from equation_constant import EquationConstant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # for importing from biological_model

from biological_model import BiologicalModel
from main_window_view import MainWindowView



class MainWindowController:
    _instance = None
    # update_ui = pyqtSignal(list)

    def __init__(self):

        self.equation_model = BiologicalModel.instance()
        self.ai_predict_model = UIUsedAIPrediction().instance()
        self.view = MainWindowView(self)
        self.equation_pred = {}
        self.equation_mask = {}
        self.real_mask = {}
        self.ai_predict_mask = {}


    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance

    # def run_equation_model(self, diffusion, reaction, grey_diff, white_diff, scan):
    #
    #     self.equation_model.set_csf_diffusion_rate(diffusion)
    #     self.equation_model.set_reaction_rate(reaction)
    #     resultFig, cur_slice_index, max_slices = self.equation_model.start_equation(scan, grey_diff, white_diff)
    #     self.view.init_sliders(cur_slice_index, max_slices)
    #     self.view.update_equation_graph(resultFig)
    #     # self.update_ui.emit(resultFig)


    def start_prediction(self, reaction, csf_diff, grey_diff, white_diff, scan, show_eq, show_real, show_ai, mixed, is_overlay):
        # TODO: Need to update for AI
        self.equation_model.set_csf_diffusion_rate(csf_diff)
        self.equation_model.set_reaction_rate(reaction)
        eq_pred, eq_mask, real_mask, cur_slice_index, max_slices = self.equation_model.start_equation(scan, grey_diff, white_diff)
        ai_mask = self.ai_predict_model.predict_using_ai(cur_slice_index)
        self.ai_predict_mask = self.reformat_data(ai_mask, True, is_ai=True)
        self.equation_pred = self.reformat_data(eq_pred, False)
        self.equation_mask = self.reformat_data(eq_mask, True)
        self.real_mask = self.reformat_data(real_mask, True)
        self.view.init_sliders(cur_slice_index, max_slices)
        self.update_image_display(show_eq, show_real, show_ai, mixed, is_overlay)

    def reformat_data(self, eq_dat, is_mask, is_ai=False):
        if is_ai:
            eq_dat = np.ascontiguousarray(eq_dat)
        else:
            for partial_data in eq_dat:
                eq_dat[partial_data] = np.flipud(eq_dat.get(partial_data))
                eq_dat[partial_data] = np.ascontiguousarray(eq_dat.get(partial_data))
                if not is_mask:
                    eq_dat[partial_data] = np.clip(eq_dat.get(partial_data) * 255, 0, 255).astype(np.uint8)
        return eq_dat

    def update_image_display(self, show_eq, show_real, show_ai, mixed, overlay):
        try:
            if self.equation_pred is None or self.equation_mask is None:
                return
            sag, cor, axi = self.update_image_color(show_eq, show_real, show_ai, mixed, overlay)
            self.view.update_plot(sag, cor, axi)
        except:
            print("Something went wrong, probably slice/time index out of range")


    def update_image_color(self, show_eq, show_real, show_ai, mixed, overlay):
        # TODO: Need to update for AI

        sag = self.equation_pred.get(EquationConstant.SAG).copy()
        cor = self.equation_pred.get(EquationConstant.COR).copy()
        axi = self.equation_pred.get(EquationConstant.AXI).copy()
        if show_eq and overlay:
            sag_eq_mask = self.equation_mask.get(EquationConstant.SAG) == 1
            sag[sag_eq_mask] = [255, 0, 0]
            cor_eq_mask = self.equation_mask.get(EquationConstant.COR) == 1
            cor[cor_eq_mask] = [255, 0, 0]
            axi_eq_mask = self.equation_mask.get(EquationConstant.AXI) == 1
            axi[axi_eq_mask] = [255, 0, 0]
        if show_real:
            sag_real_mask = self.real_mask.get(EquationConstant.SAG) == 1
            sag[sag_real_mask] = [0, 255, 0]
            cor_real_mask = self.real_mask.get(EquationConstant.COR) == 1
            cor[cor_real_mask] = [0, 255, 0]
            axi_real_mask = self.real_mask.get(EquationConstant.AXI) == 1
            axi[axi_real_mask] = [0, 255, 0]
        if show_ai:  # ai only has axi
            axi_ai_mask = self.ai_predict_mask == 1
            # axi[axi_ai_mask] = [0, 0, 255]
            axi[axi_ai_mask] = [176, 242, 70]
        if mixed and ((show_eq and overlay) or show_real or show_ai):
            if show_eq and show_real and show_ai:
                sag[sag_eq_mask & sag_real_mask] = [255, 255, 0]
                cor[cor_eq_mask & cor_real_mask] = [255, 255, 0]
                axi[axi_eq_mask & axi_real_mask & axi_ai_mask] = [255, 255, 255]
            elif show_eq and show_real:
                sag[sag_eq_mask & sag_real_mask] = [255, 255, 0]
                cor[cor_eq_mask & cor_real_mask] = [255, 255, 0]
                axi[axi_eq_mask & axi_real_mask] = [255, 255, 0]
            elif show_eq and show_ai:
                axi[axi_eq_mask & axi_ai_mask] = [255, 0, 255]
            elif show_real and show_ai:
                axi[axi_real_mask & axi_ai_mask] = [0, 255, 255]

        return sag, cor, axi

    def update_mask(self, mask1, mask2):
        returned_mask = {}
        for img_key in mask2:
            if mask1.get(img_key) is None:
                returned_mask[img_key] = mask2.get(img_key)
            else:
                returned_mask[img_key] = mask1.get(img_key) and mask2.get(img_key)

    def set_selected_file(self, file_key, file_path):
        if EquationConstant.FILE_KEYS.__contains__(file_key):
            self.equation_model.update_file_paths(file_key, file_path)
            if file_key == EquationConstant.FLAIR_KEY:
                self.ai_predict_model.set_flair1(file_path)
        else:
            self.ai_predict_model.set_flair2(file_path)

    def process_plts(self, scan, slice_i, time_i, is_overlay, show_eq, show_real, show_ai, mixed):

        eq_pred, eq_mask, real_mask = self.equation_model.update(slice_i, time_i, is_overlay, scan)
        self.ai_predict_mask = self.reformat_data(self.ai_predict_model.get_slice_prediction(slice_i), True, is_ai=True)
        self.equation_pred = self.reformat_data(eq_pred, False)
        self.equation_mask = self.reformat_data(eq_mask, True)
        self.real_mask = self.reformat_data(real_mask, True)
        self.update_image_display(show_eq, show_real, show_ai, mixed, is_overlay)

        time_day = int(self.equation_model.time_in_days(time_i))
        self.view.update_slider_value_labels(time_day)

    def save_mask(self, s, t):
        self.equation_model.save_current_mask(s, t)