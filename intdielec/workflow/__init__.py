import csv
import os
import pickle
import h5py


class Eps:
    def __init__(
        self,
        work_dir: str = None,
        data_fmt: str = "pkl",
    ) -> None:
        self.work_dir = work_dir

        self.data_fmt = data_fmt
        self._load_data()

    def _load_data(self):
        fname = os.path.join(self.work_dir, "eps_data.%s" % self.data_fmt)
        if os.path.exists(fname):
            getattr(self, "_load_data_%s" % self.data_fmt)(fname)
        else:
            self.results = {}

    def _load_data_csv(self, fname):
        with open(fname, "r") as f:
            data = csv.reader(f)
            self.results = {rows[0]: rows[1] for rows in data}

    def _load_data_pkl(self, fname):
        with open(fname, "rb") as f:
            self.results = pickle.load(f)

    def _load_data_hdf5(self, fname):
        self.results = {}

    def _save_data(self):
        fname = os.path.join(self.work_dir, "eps_data.%s" % self.data_fmt)
        getattr(self, "_save_data_%s" % self.data_fmt)(fname)

    def _save_data_csv(self, fname):
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.results.keys())
            writer.writeheader()
            writer.writerow(self.results)

    def _save_data_pkl(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.results, f)

    def _save_data_hdf5(self, fname):
        with h5py.File(fname, "a") as f:
            n = len(f)
            dts = f.create_group("%02d" % n)
            for k, v in self.results.items():
                dts.create_dataset(k, data=v)
