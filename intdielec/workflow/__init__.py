import logging
import os

from ..exts.toolbox.toolbox.utils.utils import load_dict, save_dict, safe_makedirs


class Eps:
    def __init__(
        self,
        work_dir: str = None,
        data_fmt: str = "pkl",
    ) -> None:
        self.work_dir = work_dir
        safe_makedirs(work_dir)
        self.data_fmt = data_fmt
        self._load_data()

    def workflow(self, configs: str = "param.json", default_command=None):
        logging.info("{:=^50}".format(" Start: eps Calculation "))

        self.wf_configs = load_dict(configs)
        # set env variables
        load_module = self.wf_configs.get("load_module", [])
        command = ""
        if len(load_module) > 0:
            command = "module load "
            for m in load_module:
                command += (m + " ")
            command += "&& "
        _command = self.wf_configs.get("command", default_command)
        command += _command
        # command += " && touch finished_tag"
        self.command = command
        
        default_machine_setup = {
            "remote_root": "/public/home/jxzhu/tmp_calc"
        }
        user_machine_setup = self.wf_configs.get("machine", {})
        default_machine_setup.update(user_machine_setup)
        self.machine_setup = default_machine_setup
        
        default_resources_setup = {
            "queue_name": "small_s",
            "number_node": 1,
            "cpu_per_node": 64,
            "group_size": 1,
            "module_list": [
                "gcc/9.3",
                "intel/2020.2",
                "cp2k/2022.1-intel-2020"
            ],
            "envs": {
                "OMP_NUM_THREADS": 1
            }
        }
        user_resources_setup = self.wf_configs.get("resources", {})
        default_resources_setup.update(user_resources_setup)
        self.resources_setup = default_resources_setup
        
        default_task_setup = {
            "command": "mpirun cp2k.psmp -i input.inp",
            "backward_files": [
                "output", "cp2k-RESTART.wfn",
                "cp2k-TOTAL_DENSITY-1_0.cube", "cp2k-v_hartree-1_0.cube",
            ]
        }
        user_task_setup = self.wf_configs.get("task", {})
        default_task_setup.update(user_task_setup)
        self.task_setup = default_task_setup

    def _load_data(self, fname="eps_data"):
        fname = os.path.join(self.work_dir, "%s.%s" % (fname, self.data_fmt))
        if os.path.exists(fname):
            self.results = load_dict(fname=fname)
        else:
            self.results = {}

    def _save_data(self, fname="eps_data"):
        fname = os.path.join(self.work_dir, "%s.%s" % (fname, self.data_fmt))
        # getattr(self, "_save_data_%s" % self.data_fmt)(fname)
        save_dict(d=self.results, fname=fname)
