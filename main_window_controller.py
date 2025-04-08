import sys
import os
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from UIUsedAIPrediction import UIUsedAIPrediction
from equation_constant import EquationConstant

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # for importing from biological_model

from biological_model import BiologicalModel

class MainWindowController(QThread):
    _instance = None
    initSliders = pyqtSignal(int, int)
    updatePlot = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    updateTime = pyqtSignal(int)
    stopSpinner = pyqtSignal()

    def __init__(self):
        super(MainWindowController, self).__init__()

        self.equation_model = BiologicalModel.instance()
        self.ai_predict_model = UIUsedAIPrediction().instance()
        self.equation_pred = {}
        self.equation_mask = {}
        self.real_mask = {}
        self.ai_predict_mask = {}
        self.temporal_save = {}

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = MainWindowController()
        return cls._instance

    def run(self):
        self.start_prediction(self.temporal_save[0], self.temporal_save[1], self.temporal_save[2], self.temporal_save[3],
                              self.temporal_save[4], self.temporal_save[5], self.temporal_save[6], self.temporal_save[7],
                              self.temporal_save[8], self.temporal_save[9])
        self.stopSpinner.emit()

    def set_temporal(self, reaction, csf_diff, grey_diff, white_diff, scan, show_eq, show_real, show_ai, mixed, is_overlay):
        self.temporal_save = [reaction, csf_diff, grey_diff, white_diff, scan, show_eq, show_real, show_ai, mixed, is_overlay]

    def start_prediction(self, reaction, csf_diff, grey_diff, white_diff, scan, show_eq, show_real, show_ai, mixed, is_overlay):
        self.equation_model.set_csf_diffusion_rate(csf_diff)
        self.equation_model.set_reaction_rate(reaction)
        eq_pred, eq_mask, real_mask, cur_slice_index, max_slices = self.equation_model.start_equation(scan, grey_diff, white_diff)
        ai_mask = self.ai_predict_model.predict_using_ai(cur_slice_index)
        self.ai_predict_mask = self.reformat_data(ai_mask, True)
        self.equation_pred = self.reformat_data(eq_pred, False)
        self.equation_mask = self.reformat_data(eq_mask, True)
        self.real_mask = self.reformat_data(real_mask, True)
        self.initSliders.emit(cur_slice_index, max_slices)
        self.update_image_display(show_eq, show_real, show_ai, mixed, is_overlay)

    def reformat_data(self, eq_dat, is_mask):
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
            # self.view.update_plot(sag, cor, axi)
            self.updatePlot.emit(sag, cor, axi)
        except:
            print("Something went wrong, probably slice/time index out of range")

    def detail_plots(self):
        eq_sag = self.equation_pred.get(EquationConstant.SAG).copy()
        eq_cor = self.equation_pred.get(EquationConstant.COR).copy()
        eq_axi = self.equation_pred.get(EquationConstant.AXI).copy()
        ai_sag = self.equation_pred.get(EquationConstant.SAG).copy()
        ai_cor = self.equation_pred.get(EquationConstant.COR).copy()
        ai_axi = self.equation_pred.get(EquationConstant.AXI).copy()
        #eq:
        sag_eq_mask = self.equation_mask.get(EquationConstant.SAG) == 1
        eq_sag[sag_eq_mask] = [255, 0, 0]
        cor_eq_mask = self.equation_mask.get(EquationConstant.COR) == 1
        eq_cor[cor_eq_mask] = [255, 0, 0]
        axi_eq_mask = self.equation_mask.get(EquationConstant.AXI) == 1
        eq_axi[axi_eq_mask] = [255, 0, 0]
        # ai
        sag_ai_mask = self.ai_predict_mask.get(EquationConstant.SAG) == 1
        ai_sag[sag_ai_mask] = [0, 0, 255]
        cor_ai_mask = self.ai_predict_mask.get(EquationConstant.COR) == 1
        ai_cor[cor_ai_mask] = [0, 0, 255]
        axi_ai_mask = self.ai_predict_mask.get(EquationConstant.AXI) == 1
        ai_axi[axi_ai_mask] = [0, 0, 255]

        return eq_sag, eq_cor, eq_axi, ai_sag, ai_cor, ai_axi

    def update_image_color(self, show_eq, show_real, show_ai, mixed, overlay):

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
            sag_ai_mask = self.ai_predict_mask.get(EquationConstant.SAG) == 1
            sag[sag_ai_mask] = [0, 0, 255]
            cor_ai_mask = self.ai_predict_mask.get(EquationConstant.COR) == 1
            cor[cor_ai_mask] = [0, 0, 255]
            axi_ai_mask = self.ai_predict_mask.get(EquationConstant.AXI) == 1
            axi[axi_ai_mask] = [0, 0, 255]
        if mixed and ((show_eq and overlay) or show_real or show_ai):
            if (show_eq and overlay) and show_real and show_ai:
                sag[sag_eq_mask & sag_real_mask & sag_ai_mask] = [255, 255, 255]
                cor[cor_eq_mask & cor_real_mask & cor_ai_mask] = [255, 255, 255]
                axi[axi_eq_mask & axi_real_mask & axi_ai_mask] = [255, 255, 255]
            elif (show_eq and overlay) and show_real:
                sag[sag_eq_mask & sag_real_mask] = [255, 255, 0]
                cor[cor_eq_mask & cor_real_mask] = [255, 255, 0]
                axi[axi_eq_mask & axi_real_mask] = [255, 255, 0]
            elif (show_eq and overlay) and show_ai:
                color = [255,204,255]
                axi[axi_eq_mask & axi_ai_mask] = color
                sag[sag_eq_mask & sag_ai_mask] = color
                cor[cor_eq_mask & cor_ai_mask] = color
            elif show_real and show_ai:
                axi[axi_real_mask & axi_ai_mask] = [0, 255, 255]
                sag[sag_real_mask & sag_ai_mask] = [0, 255, 255]
                cor[cor_real_mask & cor_ai_mask] = [0, 255, 255]
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

    def process_plots(self, scan, slice_i, time_i, is_overlay, show_eq, show_real, show_ai, mixed):
        eq_pred, eq_mask, real_mask = self.equation_model.update(slice_i, time_i, is_overlay, scan)
        self.ai_predict_mask = self.reformat_data(self.ai_predict_model.get_slice_prediction(slice_i), True)
        self.equation_pred = self.reformat_data(eq_pred, False)
        self.equation_mask = self.reformat_data(eq_mask, True)
        self.real_mask = self.reformat_data(real_mask, True)
        self.update_image_display(show_eq, show_real, show_ai, mixed, is_overlay)

        time_day = int(self.equation_model.time_in_days(time_i))
        self.updateTime.emit(time_day)


    def save_mask(self, s, t):
        self.equation_model.save_current_mask(s, t)