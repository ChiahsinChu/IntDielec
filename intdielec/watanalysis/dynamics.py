import numpy as np
from MDAnalysis.analysis.waterdynamics import (MeanSquareDisplacement,
                                               SurvivalProbability,
                                               WaterOrientationalRelaxation)
import logging
from ..utils.mda import make_selection


# TODO: fix the slice analysis
class MSD(MeanSquareDisplacement):
    def __init__(self, universe, t0, tf, dtmax, nproc=1, axis=2, **kwargs):
        select = make_selection(**kwargs)
        logging.info("selection: %s" % select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)
        self.axis = axis

    def _getOneDeltaPoint(self, universe, repInd, i, t0, dt):
        val_para = 0
        val_perp = 0

        n = 0
        for j in range(len(repInd[i]) // 3):
            begj = 3 * j
            universe.trajectory[t0]
            # Plus zero is to avoid 0to be equal to 0tp
            Ot0 = repInd[i][begj].position + 0

            universe.trajectory[t0 + dt]
            # Plus zero is to avoid 0to be equal to 0tp
            Otp = repInd[i][begj].position + 0

            # position oxygen
            OVector = Ot0 - Otp
            # here it is the difference with
            # waterdynamics.WaterOrientationalRelaxation
            val_perp += np.square(OVector[self.axis])
            val_para += (np.dot(OVector, OVector) -
                         np.square(OVector[self.axis]))
            # valO += np.dot(OVector, OVector)
            n += 1

        # if no water molecules remain in selection, there is nothing to get
        # the mean, so n = 0.
        return val_perp / n if n > 0 else 0, val_para / n if n > 0 else 0

    def _getMeanOnePoint(self, universe, selection1, dt, totalFrames):
        """
        This function gets one point of the plot C_vec vs t. It's uses the
        _getOneDeltaPoint() function to calculate the average.

        """
        repInd = self._repeatedIndex(selection1, dt, totalFrames)
        sumsdt = 0
        n = 0.0
        sumDeltaO_perp = 0.0
        sumDeltaO_para = 0.0
        # valOList_perp = []
        # valOList_para = []

        for j in range(totalFrames // dt - 1):
            a_perp, a_para = self._getOneDeltaPoint(universe, repInd, j,
                                                    sumsdt, dt)
            sumDeltaO_perp += a_perp
            sumDeltaO_para += a_para
            # valOList_perp.append(a_perp)
            # valOList_para.append(a_para)
            sumsdt += dt
            n += 1

        # if no water molecules remain in selection, there is nothing to get
        # the mean, so n = 0.
        return sumDeltaO_perp / n if n > 0 else 0, sumDeltaO_para / n if n > 0 else 0

    def run(self, **kwargs):
        """Analyze trajectory and produce timeseries"""

        # All the selection to an array, this way is faster than selecting
        # later.
        if self.nproc == 1:
            selection_out = self._selection_serial(self.universe,
                                                   self.selection)
        else:
            # parallel not yet implemented
            # selection = selection_parallel(universe, selection_str, nproc)
            selection_out = self._selection_serial(self.universe,
                                                   self.selection)
        self.timeseries_perp = []
        self.timeseries_para = []
        for dt in list(range(1, self.dtmax + 1)):
            output_perp, output_para = self._getMeanOnePoint(
                self.universe, selection_out, dt, self.tf)
            self.timeseries_perp.append(output_perp)
            self.timeseries_para.append(output_para)
        self.timeseries = np.array(self.timeseries_para) + np.array(
            self.timeseries_perp)


class SP(SurvivalProbability):
    def __init__(self, universe, verbose=False, **kwargs):
        select = make_selection(**kwargs)
        logging.info("selection: %s" % select)
        super().__init__(universe, select, verbose)


class WOR(WaterOrientationalRelaxation):
    def __init__(self, universe, t0, tf, dtmax, nproc=1, **kwargs):
        """
        sel_region, surf_ids, c_ag, select_all, bonded
        """
        select = make_selection(**kwargs)
        logging.info("selection: %s" % select)
        super().__init__(universe, select, t0, tf, dtmax, nproc)
