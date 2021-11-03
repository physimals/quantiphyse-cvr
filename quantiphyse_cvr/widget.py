"""
CVR Quantiphyse plugin

Author: Martin Craig <martin.craig@nottingham.ac.uk>
Copyright (c) 2021 University of Nottingham, Martin Craig
"""

from __future__ import division, unicode_literals, absolute_import, print_function

from PySide2 import QtGui, QtCore, QtWidgets

from quantiphyse.gui.widgets import QpWidget, Citation, TitleWidget, RunWidget
from quantiphyse.gui.options import OptionBox, DataOption, NumericOption, BoolOption, NumberListOption, TextOption, FileOption, ChoiceOption
from quantiphyse.utils import LogSource

from ._version import __version__

FAB_CITE_TITLE = "Variational Bayesian inference for a non-linear forward model"
FAB_CITE_AUTHOR = "Chappell MA, Groves AR, Whitcher B, Woolrich MW."
FAB_CITE_JOURNAL = "IEEE Transactions on Signal Processing 57(1):223-236, 2009."

class OptionsWidget(QtWidgets.QWidget, LogSource):

    sig_changed = QtCore.Signal()

    def __init__(self, ivm, parent):
        LogSource.__init__(self)
        QtWidgets.QWidget.__init__(self, parent)
        self.ivm = ivm

class AcquisitionOptions(OptionsWidget):

    N_REGRESSORS = 3

    def __init__(self, ivm, parent):
        OptionsWidget.__init__(self, ivm, parent)

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        self._optbox = OptionBox()
        self._optbox.add("<b>Data</b>")
        self._optbox.add("BOLD timeseries data", DataOption(self.ivm), key="data")
        self._optbox.add("ROI", DataOption(self.ivm, rois=True, data=False), key="roi")

        #self._optbox.add("Physiological data (CO<sub>2</sub>/O<sub>2</sub>)", FileOption(plot_btn=True), key="phys-data")
        #self._optbox.add("Sampling frequency (Hz)", NumericOption(minval=0, maxval=1000, default=100, intonly=True), key="samp-rate")
        self._optbox.add("TR for MRI timeseries (s)", NumericOption(minval=0, maxval=5, default=1.0), key="tr")
        self._optbox.add("Baseline period (s)", NumericOption(minval=0, maxval=200, default=60, intonly=True), key="baseline")
        self._optbox.add("MRI timeseries alignment", ChoiceOption(["Automatic", "Manual"]), key="mri-align")
        self._optbox.option("mri-align").sig_changed.connect(self._align_changed)
        self._optbox.add("MRI timeseries start time (s)", NumericOption(minval=0, maxval=1000, default=0), key="data-start-time")
        vbox.addWidget(self._optbox)

        self._optbox_reg = OptionBox()
        self._optbox_reg.add("<b>Regressors</b>")
        for idx in range(self.N_REGRESSORS):
            self._optbox_reg.add("Regressor %i" % (idx+1), ChoiceOption(["Unprocessed CO2", "Preprocessed pETCO2", "Ramp (linear drift)", "Custom"], ["co2", "petco2", "ramp", "custom"]), 
                             checked=True, default=True, key="type_%i" % (idx+1))
            self._optbox_reg.option("type_%i" % (idx+1)).sig_changed.connect(self._regressor_changed)
            self._optbox_reg.add("Data (CO<sub>2</sub>/O<sub>2</sub>)", FileOption(plot_btn=True), key="data_%i" % (idx+1))
            self._optbox_reg.add("Time resolution (s)", NumericOption(minval=0, maxval=10, default=1), key="tr_%i" % (idx+1))

        vbox.addWidget(self._optbox_reg)

        vbox.addStretch(1)
        self._regressor_changed()
        self._align_changed()

    def _regressor_changed(self):
        for idx in range(self.N_REGRESSORS):
            opts = self._optbox_reg.values()
            extras_visible = "type_%i" % (idx+1) in opts and opts["type_%i" % (idx+1)] != "ramp"
            self._optbox_reg.set_visible("data_%i" % (idx+1), extras_visible)
            self._optbox_reg.set_visible("tr_%i" % (idx+1), extras_visible)

    def _add_regressor_options(self, opts):
        regressors = []
        regressor_types = []
        regressor_trs = []
        reg_opts = self._optbox_reg.values()
        for idx in range(self.N_REGRESSORS):
            regressor_type = reg_opts.get("type_%i" % (idx+1), None)
            if regressor_type is not None:
                if regressor_type != "ramp":
                    regressor_types.append(regressor_type)
                    regressors.append(reg_opts["data_%i" % (idx+1)])
                    regressor_trs.append(reg_opts["tr_%i" % (idx+1)])
                else:
                    # FIXME can't mix file regressors with Numpy array, need to write to tmp file
                    regressor_types.append("custom")
                    regressor_trs.append(opts["tr"])
                    regressors.append(np.linspace(0, 1, self.ivm.data[opts["data"]].nvols))
        opts["regressors"] = ",".join(regressors)
        opts["regressor_trs"] = ",".join(["%.3f" % v for v in regressor_trs])
        opts["regressor_types"] = ",".join(regressor_types)

    def _align_changed(self):
        self._optbox.set_visible("data-start-time", self._optbox.option("mri-align").value == "Manual")

    def options(self):
        opts = self._optbox.values()
        opts.pop("mri-align", None)
        self._add_regressor_options(opts)
        return opts

class FabberVbOptions(OptionsWidget):
    def __init__(self, ivm, parent, acq_options):
        OptionsWidget.__init__(self, ivm, parent)
        self.acq_options = acq_options

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        cite = Citation(FAB_CITE_TITLE, FAB_CITE_AUTHOR, FAB_CITE_JOURNAL)
        vbox.addWidget(cite)

        self._optbox = OptionBox()
        self._optbox.add("<b>Model options</b>")
        self._optbox.add("Infer constant signal offset", BoolOption(default=True), key="infer-sig0")
        self._optbox.add("Infer delay", BoolOption(default=True), key="infer-delay")

        #self._optbox.add("<b>Model fitting options</b>")
        #self._optbox.add("Spatial regularization", BoolOption(default=True), key="spatial")
        self._optbox.add("<b>Output options</b>")
        self._optbox.add("Output data name suffix", TextOption(), checked=True, key="output-suffix")

        vbox.addWidget(self._optbox)
        vbox.addWidget(RunWidget(self, save_option=True))
        vbox.addStretch(1)

    def processes(self):
        opts = {
            "model-group" : "cvr",
            "model" : "cvr_petco2",
            "save-mean" : True,
            "save-model-fit" : True,
            "noise" : "white",
            "max-iterations" : 10,
        }
        opts.update(self.acq_options.options())
        opts.update(self._optbox.values())

        # Fabber model requires the physiological data to be preprocessed
        from vaby.data import DataModel
        data_model = DataModel(opts["data"].raw(), mask=opts["mask"].raw())
        from vaby_models_cvr.petco2 import CvrPetCo2Model
        opts["phys_data"] = opts["phys-data"] # FIXME hack
        model = CvrPetCo2Model(data_model, **opts)
        opts["phys-data"] = model.co2_mmHg
        opts.pop("phys_data")

        # Deal with the output suffix if specified
        suffix = opts.pop("output-suffix", "")
        if suffix and suffix[0] != "_":
            suffix = "_" + suffix
        opts["output-rename"] = {
                "mean_cvr" : "cvr%s" % suffix,
                "mean_sig0" : "sig0%s" % suffix,
                "mean_delay" : "delay%s" % suffix,
                "modelfit" : "modelfit%s" % suffix,
        }

        # In spatial mode use sig0 as regularization parameter
        if opts.pop("spatial", False):
            opts["method"] = "spatialvb"
            opts["param-spatial-priors"] = "M+"

        self.debug("Fabber CVR options: %s", opts)
        processes = [
            {"Fabber" : opts},
        ]

        return processes

class VbOptions(OptionsWidget):
    def __init__(self, ivm, parent, acq_options):
        OptionsWidget.__init__(self, ivm, parent)
        self.acq_options = acq_options

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        cite = Citation(FAB_CITE_TITLE, FAB_CITE_AUTHOR, FAB_CITE_JOURNAL)
        vbox.addWidget(cite)

        self._optbox = OptionBox()
        self._optbox.add("<b>Model options</b>")
        self._optbox.add("Infer constant signal offset", BoolOption(default=True), key="infer-sig0")
        self._optbox.add("Infer delay", BoolOption(default=True), key="infer-delay")

        self._optbox.add("<b>Model fitting options</b>")
        self._optbox.add("Number of iterations", NumericOption(minval=0, maxval=100, default=10, intonly=True), key="max-iterations")
        #self._optbox.add("Spatial regularization", BoolOption(default=True), key="spatial")

        self._optbox.add("<b>Output options</b>")
        self._optbox.add("Output variance maps", BoolOption(), key="output-var")
        self._optbox.add("Output data name suffix", TextOption(), checked=True, key="output-suffix")

        vbox.addWidget(self._optbox)
        vbox.addWidget(RunWidget(self, save_option=True))
        vbox.addStretch(1)

    def processes(self):
        opts = {}
        opts.update(self.acq_options.options())
        opts.update(self._optbox.values())

        self.debug("CvrPetCo2Vb options: %s", opts)
        processes = [
            {"CvrPetCo2Vb" : opts},
        ]

        return processes

class GlmOptions(OptionsWidget):
    def __init__(self, ivm, parent, acq_options):
        OptionsWidget.__init__(self, ivm, parent)
        self.acq_options = acq_options

        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        self._optbox = OptionBox()
        self._optbox.add("<b>Model options</b>")
        self._optbox.add("Delay minimum (s)", NumericOption(minval=-100, maxval=100, default=0), key="delay-min")
        self._optbox.add("Delay maximum (s)", NumericOption(minval=-100, maxval=100, default=0), key="delay-max")
        self._optbox.add("Delay step (s)", NumericOption(minval=-5, maxval=5, default=1), key="delay-step")

        self._optbox.add("<b>Output options</b>")
        self._optbox.add("Output data name suffix", TextOption(), checked=True, key="output-suffix")

        vbox.addWidget(self._optbox)
        vbox.addWidget(RunWidget(self, save_option=True))
        vbox.addStretch(1)

    def processes(self):
        opts = {}
        opts.update(self.acq_options.options())
        opts.update(self._optbox.values())
        self.debug("CvrPetCo2Glm options: %s", opts)
        processes = [
            {"CvrPetCo2Glm" : opts},
        ]

        return processes

class CvrPetCo2Widget(QpWidget):
    """
    CVR modelling of BOLD-MRI with PETCO2
    """
    def __init__(self, **kwargs):
        QpWidget.__init__(self, name="CVR PETCO2", icon="cvr", group="CVR",
                          desc="Cerebrovascular reactivity using BOLD-MRI and PETCO2", **kwargs)
        self.current_tab = 0

    def init_ui(self):
        vbox = QtWidgets.QVBoxLayout()
        self.setLayout(vbox)

        title = TitleWidget(self, help="cvr", subtitle="Cerebrovascular reactivity using BOLD-MRI and PETCO2 v%s" % __version__)
        vbox.addWidget(title)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        vbox.addWidget(self.tabs)

        self.acquisition_opts = AcquisitionOptions(self.ivm, parent=self)
        self.tabs.addTab(self.acquisition_opts, "Acquisition Options")
        #self.fabber_opts = FabberVbOptions(self.ivm, self, self.acquisition_opts)
        #self.tabs.addTab(self.fabber_opts, "Bayesian modelling")
        self.vb_opts = VbOptions(self.ivm, self, self.acquisition_opts)
        self.tabs.addTab(self.vb_opts, "Bayesian modelling")
        self.glm_opts = GlmOptions(self.ivm, self, self.acquisition_opts)
        self.tabs.addTab(self.glm_opts, "GLM modelling")

        vbox.addStretch(1)

    def _tab_changed(self):
        tab = self.tabs.currentIndex()
        if tab in (1, 2):
            self.current_tab = tab

    def processes(self):
        # For batch options, return whichever tab was last selected
        # (default to VB options if on acquisition tab)
        if self.current_tab == 1:
            return self.vb_opts.processes()
        elif self.current_tab == 2:
            return self.glm_opts.processes()
        else:
            return []
