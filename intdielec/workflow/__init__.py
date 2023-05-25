import logging
import os

from ..exts.toolbox.toolbox.utils.utils import load_dict, save_dict


class Eps:

    def __init__(
        self,
        work_dir: str = None,
        data_fmt: str = "pkl",
    ) -> None:
        self.work_dir = work_dir

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
