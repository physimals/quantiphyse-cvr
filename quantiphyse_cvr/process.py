"""
CVR Quantiphyse plugin - Processes

Author: Martin Craig <martin.craig@nottingham.ac.uk>
Copyright (c) 2021 University of Nottingham, Martin Craig
"""
import io
import logging
import sys
import os

import numpy as np

# Silence Tensorflow random messages
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

from quantiphyse.utils import QpException
from quantiphyse.utils.enums import Boundary
from quantiphyse.processes import Process

MAX_LOG_SIZE=100000

def _get_progress_cb(worker_id, queue, n_voxels):
    def _progress(frac):
        queue.put((worker_id, frac * n_voxels))
    return _progress

def _run_glm(worker_id, queue, data, mask, phys_data, tr, baseline, samp_rate, data_start_time, delay_min, delay_max, delay_step):
    try:
        from vaby.data import DataModel
        from vaby_models_cvr.petco2 import CvrPetCo2Model

        options = {
            "phys_data" : phys_data,
            "tr" : tr,
            "baseline" : baseline,
            #"blocksize_on" : blocksize_on,
            #"blocksize_off" : blocksize_off,
            "samp_rate" : samp_rate,
            "data_start_time" : data_start_time,
            #"delay" : mech_delay,
        }

        # Set up log to go to string buffer
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        data_model = DataModel(data, mask=mask)
        fwd_model = CvrPetCo2Model(data_model, **options)
        cvr, delay, sig0, modelfit = fwd_model.fit_glm(delay_min=delay_min, delay_max=delay_max, delay_step=delay_step, progress_cb=_get_progress_cb(worker_id, queue, data_model.n_unmasked_voxels))
        ret = {"cvr" : cvr, "delay" : delay, "sig0" : sig0, "modelfit" : modelfit}
        for name in list(ret.keys()):
            data = ret[name]
            shape = data_model.shape
            if data.ndim > 1:
                shape = list(shape) + [data.shape[1]]
            ndata = np.zeros(shape, dtype=np.float)
            ndata[mask > 0] = data
            ret[name] = ndata
        queue.put((worker_id, data_model.n_unmasked_voxels))

        ret = (ret, log.getvalue())
        return worker_id, True, ret
    except:
        import traceback
        traceback.print_exc()
        return worker_id, False, sys.exc_info()[1]

class CvrPetCo2GlmProcess(Process):
    """
    CVR-PETCO2 modelling using a GLM
    """
    
    PROCESS_NAME = "CvrPetCo2Glm"
    
    def __init__(self, ivm, **kwargs):
        Process.__init__(self, ivm, worker_fn=_run_glm, **kwargs)

    def run(self, options):
        data = self.get_data(options)
        if data.ndim != 4: 
            raise QpException("Data must be 4D for DCE PK modelling")

        roi = self.get_roi(options, data.grid)
    
        self.suffix = options.pop('output-suffix', '')
        if self.suffix != "" and self.suffix[0] != "_": 
            self.suffix = "_" + self.suffix

        phys_data = options.pop('phys-data', None)
        if phys_data is None:
            raise QpException("Physiological data option 'phys-data' must be given")
        if isinstance(phys_data, str) and not os.path.isabs(phys_data):
            phys_data = os.path.join(self.indir, phys_data)
        tr = options.pop("tr", None)
        if tr is None:
            raise QpException("TR must be given")

        # Non-compulsary options
        baseline = options.pop("baseline", 60)
        #blocksize_on = options.pop("blocksize-on", 120)
        #blocksize_off = options.pop("blocksize-off", 120)
        samp_rate = options.pop("samp-rate", 100)
        #mech_delay = options.pop("delay", 15)
        data_start_time = options.pop("data-start-time", None)
        delay_min = options.pop("delay-min", 0)
        delay_max = options.pop("delay-max", 0)
        delay_step = options.pop("delay-step", 1)
        
        # Use smallest sub-array of the data which contains all unmasked voxels
        self.grid = data.grid
        self.bb_slices = roi.get_bounding_box()
        self.debug("Using bounding box: %s", self.bb_slices)
        data_bb = data.raw()[tuple(self.bb_slices)]
        mask_bb = roi.raw()[tuple(self.bb_slices)]
        #n_workers = data_bb.shape[0]
        n_workers = 1

        args = [data_bb, mask_bb, phys_data, tr, baseline, samp_rate, data_start_time, delay_min, delay_max, delay_step]
        self.voxels_done = [0] * n_workers
        self.total_voxels = np.count_nonzero(roi.raw())
        self.start_bg(args, n_workers=n_workers)

    def timeout(self, queue):
        if queue.empty(): return
        while not queue.empty():
            worker_id, voxels_done = queue.get()
            self.voxels_done[worker_id] = voxels_done
        progress = float(sum(self.voxels_done)) / self.total_voxels
        self.sig_progress.emit(progress)

    def finished(self, worker_output):
        """
        Add output data to the IVM
        """
        if self.status == Process.SUCCEEDED:
            # Only include log from first process to avoid multiple repetitions
            for data, log in worker_output:
                data_keys = data.keys()
                if log:
                    # If there was a problem the log could be huge and full of
                    # nan messages. So chop it off at some 'reasonable' point
                    self.log(log[:MAX_LOG_SIZE])
                    if len(log) > MAX_LOG_SIZE:
                        self.log("WARNING: Log was too large - truncated at %i chars" % MAX_LOG_SIZE)
                    break
            first = True
            self.data_items = []
            for key in data_keys:
                self.debug(key)
                recombined_data = self.recombine_data([o.get(key, None) for o, l in worker_output])
                name = key + self.suffix
                if key is not None:
                    self.data_items.append(name)
                    if recombined_data.ndim == 2:
                        recombined_data = np.expand_dims(recombined_data, 2)

                    # The processed data was chopped out of the full data set to just include the
                    # ROI - so now we need to put it back into a full size data set which is otherwise
                    # zero.
                    if recombined_data.ndim == 4:
                        shape4d = list(self.grid.shape) + [recombined_data.shape[3],]
                        full_data = np.zeros(shape4d, dtype=np.float32)
                    else:
                        full_data = np.zeros(self.grid.shape, dtype=np.float32)
                    full_data[self.bb_slices] = recombined_data.reshape(full_data[self.bb_slices].shape)
                    self.ivm.add(full_data, grid=self.grid, name=name, make_current=first, roi=False)

                    # Set some view defaults because we know what range these should be in
                    self.ivm.data[name].view.boundary = Boundary.CLAMP
                    if key == "cvr":
                        self.ivm.data[name].view.cmap_range = (0, 1)
                    if key == "delay":
                        self.ivm.data[name].view.cmap_range = (-15, 15)

                    first = False
        else:
            # Include the log of the first failed process
            for out in worker_output:
                if out and isinstance(out, Exception) and hasattr(out, "log") and len(out.log) > 0:
                    self.log(out.log)
                    break

def _run_vb(worker_id, queue, data, mask, phys_data, tr, infer_sig0, infer_delay, baseline, samp_rate, data_start_time, spatial, maxits, output_var):
    try:
        from vaby.data import DataModel
        from vaby_avb import Avb
        from vaby_models_cvr.petco2 import CvrPetCo2Model

        options = {
            "phys_data" : phys_data,
            "tr" : tr,
            "baseline" : baseline,
            #"blocksize_on" : blocksize_on,
            #"blocksize_off" : blocksize_off,
            "samp_rate" : samp_rate,
            "data_start_time" : data_start_time,
            #"delay" : mech_delay,
            "infer_sig0" : infer_sig0,
            "infer_delay" : infer_delay,
            "max_iterations" : maxits,
        }

        if spatial:
            options["param_overrides"] = {}
            for param in ("cvr", "delay", "sig0"):
                options["param_overrides"][param] = {"prior_type" : "M"}

        data_model = DataModel(data, mask=mask, **options)
        fwd_model = CvrPetCo2Model(data_model, **options)

        # Set up log to go to string buffer
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        tpts = fwd_model.tpts()
        avb = Avb(tpts, data_model, fwd_model, progress_cb=_get_progress_cb(worker_id, queue, data_model.n_unmasked_voxels), **options)
        avb.run()

        ret = {}
        for idx, param in enumerate(fwd_model.params):
            data = avb.model_mean[idx]
            ret[param.name] = data_model.nifti_image(data).get_fdata()
            if output_var:
                data = avb.model_var[idx]
                ret[param.name + "_var"] = data_model.nifti_image(data).get_fdata()

        ret["modelfit"] = data_model.nifti_image(avb.modelfit).get_fdata()
        ret = (ret, log.getvalue())

        queue.put((worker_id, data_model.n_unmasked_voxels))
        return worker_id, True, ret
    except:
        import traceback
        traceback.print_exc()
        return worker_id, False, sys.exc_info()[1]

class CvrPetCo2VbProcess(Process):
    """
    CVR-PETCO2 modelling using VB
    """

    PROCESS_NAME = "CvrPetCo2Vb"

    def __init__(self, ivm, **kwargs):
        Process.__init__(self, ivm, worker_fn=_run_vb, **kwargs)

    def run(self, options):
        data = self.get_data(options)
        if data.ndim != 4:
            raise QpException("Data must be 4D for CVR modelling")

        roi = self.get_roi(options, data.grid)
    
        self.suffix = options.pop('output-suffix', '')
        if self.suffix != "" and self.suffix[0] != "_": 
            self.suffix = "_" + self.suffix

        phys_data = options.pop('phys-data', None)
        if phys_data is None:
            raise QpException("Physiological data option 'phys-data' must be given")
        if isinstance(phys_data, str) and not os.path.isabs(phys_data):
            phys_data = os.path.join(self.indir, phys_data)
        tr = options.pop("tr", None)
        if tr is None:
            raise QpException("TR must be given")

        # Non-compulsary options
        baseline = options.pop("baseline", 60)
        #blocksize_on = options.pop("blocksize-on", 120)
        #blocksize_off = options.pop("blocksize-off", 120)
        samp_rate = options.pop("samp-rate", 100)
        #mech_delay = options.pop("delay", 15)
        data_start_time = options.pop("data-start-time", None)
        spatial = options.pop("spatial", False)
        maxits = options.pop("max-iterations", 10)
        output_var = options.pop("output-var", False)

        infer_sig0 = options.pop("infer-sig0", True)
        infer_delay = options.pop("infer-delay", True)
        
        # Use smallest sub-array of the data which contains all unmasked voxels
        self.grid = data.grid
        self.bb_slices = roi.get_bounding_box()
        self.debug("Using bounding box: %s", self.bb_slices)
        data_bb = data.raw()[tuple(self.bb_slices)]
        mask_bb = roi.raw()[tuple(self.bb_slices)]
        #n_workers = data_bb.shape[0]
        n_workers = 1

        args = [data_bb, mask_bb, phys_data, tr, infer_sig0, infer_delay, baseline, samp_rate, data_start_time, spatial, maxits, output_var]
        self.voxels_done = [0] * n_workers
        self.total_voxels = np.count_nonzero(roi.raw())
        self.start_bg(args, n_workers=n_workers)

    def timeout(self, queue):
        if queue.empty(): return
        while not queue.empty():
            worker_id, voxels_done = queue.get()
            self.voxels_done[worker_id] = voxels_done
        progress = float(sum(self.voxels_done)) / self.total_voxels
        self.sig_progress.emit(progress)

    def finished(self, worker_output):
        """
        Add output data to the IVM
        """
        if self.status == Process.SUCCEEDED:
            # Only include log from first process to avoid multiple repetitions
            for data, log in worker_output:
                data_keys = data.keys()
                if log:
                    # If there was a problem the log could be huge and full of 
                    # nan messages. So chop it off at some 'reasonable' point
                    self.log(log[:MAX_LOG_SIZE])
                    if len(log) > MAX_LOG_SIZE:
                        self.log("WARNING: Log was too large - truncated at %i chars" % MAX_LOG_SIZE)
                    break
            first = True
            self.data_items = []
            for key in data_keys:
                self.debug(key)
                recombined_data = self.recombine_data([o.get(key, None) for o, l in worker_output])
                name = key + self.suffix
                if key is not None:
                    self.data_items.append(name)
                    if recombined_data.ndim == 2:
                        recombined_data = np.expand_dims(recombined_data, 2)

                    # The processed data was chopped out of the full data set to just include the
                    # ROI - so now we need to put it back into a full size data set which is otherwise
                    # zero.
                    if recombined_data.ndim == 4:
                        shape4d = list(self.grid.shape) + [recombined_data.shape[3],]
                        full_data = np.zeros(shape4d, dtype=np.float32)
                    else:
                        full_data = np.zeros(self.grid.shape, dtype=np.float32)
                    full_data[self.bb_slices] = recombined_data.reshape(full_data[self.bb_slices].shape)
                    self.ivm.add(full_data, grid=self.grid, name=name, make_current=first, roi=False)

                    # Set some view defaults because we know what range these should be in
                    self.ivm.data[name].view.boundary = Boundary.CLAMP
                    if key == "cvr":
                        self.ivm.data[name].view.cmap_range = (0, 1)
                    if key == "delay":
                        self.ivm.data[name].view.cmap_range = (-15, 15)

                    first = False
        else:
            # Include the log of the first failed process
            for out in worker_output:
                if out and isinstance(out, Exception) and hasattr(out, "log") and len(out.log) > 0:
                    self.log(out.log)
                    break
            