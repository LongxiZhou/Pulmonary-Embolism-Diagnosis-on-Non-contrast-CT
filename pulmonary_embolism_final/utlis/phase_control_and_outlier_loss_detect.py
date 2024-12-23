import numpy as np


class OutlierLossDetect:
    def __init__(self, store_count=30, remove_max_count=3, remove_min_count=3, std_outlier=10, mute=False):
        self.recent_loss_history = []
        self.store_count = store_count
        self.remove_max_count = remove_max_count
        self.remove_min_count = remove_min_count
        self.std_outlier = std_outlier

        self.in_queue_count = 0

        self.consecutive_outlier = 0

        self.mute = mute

    def update_new_loss(self, new_loss):
        # return True for non-outlier loss

        if self.mute:
            return True

        if self.consecutive_outlier > 20:
            print("consecutive outlier detected!")
            return "consecutive_outlier"

        if len(self.recent_loss_history) < self.store_count:
            self.recent_loss_history.append(new_loss)
            self.in_queue_count += 1
            self.consecutive_outlier = 0
            return True

        std_in_queue, ave_in_queue = self.get_std_and_ave_in_queue()
        lower_bound = ave_in_queue - self.std_outlier * std_in_queue
        upper_bound = ave_in_queue + self.std_outlier * std_in_queue

        if new_loss < lower_bound or new_loss > upper_bound:
            print("outlier loss:", new_loss)
            print("average recent", len(self.recent_loss_history),
                  "loss:", ave_in_queue, "std for recent", len(self.recent_loss_history), "loss:", std_in_queue)
            self.consecutive_outlier += 1
            return False

        self.recent_loss_history[self.in_queue_count % self.store_count] = new_loss
        self.in_queue_count += 1
        self.consecutive_outlier = 0
        return True

    def reset(self):
        self.in_queue_count = 0
        self.recent_loss_history = []
        self.consecutive_outlier = 0

    def get_std_and_ave_in_queue(self):

        if len(self.recent_loss_history) < self.store_count:
            std_in_queue = np.std(self.recent_loss_history)
            ave_in_queue = np.average(self.recent_loss_history)
            return std_in_queue, ave_in_queue

        temp_list = list(self.recent_loss_history)
        temp_list.sort()
        std_in_queue = np.std(temp_list[self.remove_min_count: -self.remove_max_count])
        ave_in_queue = np.average(temp_list[self.remove_min_count: -self.remove_max_count])

        return std_in_queue, ave_in_queue


class TrainingPhaseControl:
    def __init__(self, params):

        self.target_recall = params["target_recall"]
        self.target_precision = params["target_precision"]

        self.flip_recall = params["flip_recall"]
        self.flip_precision = params["flip_precision"]

        self.base_recall = params["base_recall"]
        self.base_precision = params["base_precision"]

        self.current_phase = 'warm_up'
        # 'warm_up', 'recall_phase', 'precision_phase', 'converge_to_recall', 'converge_to_precision'

        self.flip_remaining = params["flip_remaining"]
        # one flip means change the phase 'precision_phase' -> 'recall_phase'

        self.base_relative = params["base_relative"]
        # will not flip util number times recall/precision bigger than precision/recall >= base_relative

        self.max_performance_recall = params["max_performance_recall"]
        self.max_performance_precision = params["max_performance_precision"]
        # force flip when precision/recall > max_performance during precision/recall phase

        self.final_phase = params["final_phase"]  # 'converge_to_recall', 'converge_to_precision'

        self.warm_up_epochs = params["warm_up_epochs"]

        self.previous_phase = None
        self.changed_phase_in_last_epoch = False

        # --------------------------
        # check correctness
        assert 0 <= self.flip_recall <= 1 and 0 <= self.flip_precision <= 1
        assert 0 <= self.base_recall <= 1 and 0 <= self.base_precision <= 1
        assert self.flip_remaining >= 0
        assert self.warm_up_epochs >= 0

        assert self.final_phase in ['converge_to_recall', 'converge_to_precision']
        if self.final_phase == 'converge_to_recall':
            assert 0 < self.target_recall < 1
        if self.final_phase == 'converge_to_precision':
            assert 0 < self.target_precision < 1

        self.precision_to_recall_during_converging = 4
        # the precision and recall will fluctuate around the target performance. When this value to 0, end to training.

        self.epoch_passed = 0
        self.relative_false_positive_penalty = params["initial_relative_false_positive_penalty"]
        # higher means model give less false positives, at the expense of more false negative

        self.history_relative_false_positive_penalty = []
        self.history_recall = []
        self.history_precision = []

    def get_new_relative_false_positive_penalty(self, current_recall, current_precision):
        self._update_history(current_recall, current_precision)
        self.changed_phase_in_last_epoch = self._update_phase(current_recall, current_precision)
        self._update_relative_false_positive_penalty(current_recall, current_precision)
        self.show_status(current_recall, current_precision)
        self.epoch_passed += 1
        return self.relative_false_positive_penalty

    def _update_history(self, current_recall, current_precision):
        self.history_relative_false_positive_penalty.append(self.relative_false_positive_penalty)
        self.history_recall.append(current_recall)
        self.history_precision.append(current_precision)

    def _update_phase(self, current_recall, current_precision):
        # return True for phase change

        if self.previous_phase is None:
            self.previous_phase = self.current_phase  # update previous phase when update current phase

        if self.current_phase == self.final_phase:  # do not update
            return False

        if self.epoch_passed < self.warm_up_epochs:
            self.current_phase = 'warm_up'
            return False

        if self.current_phase == 'warm_up' and self.epoch_passed >= self.warm_up_epochs:
            self.current_phase = 'recall_phase'
            if (current_recall > self.flip_recall and current_recall / (
                    current_precision + 1e-8) > self.base_relative) \
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                self.previous_phase = self.current_phase
                self.current_phase = 'precision_phase'
            print("changing current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
            return True

        if self.current_phase == 'recall_phase':
            if (current_recall > self.flip_recall and current_recall / (
                    current_precision + 1e-8) > self.base_relative) \
                    or current_precision < self.base_precision or current_recall > self.max_performance_recall:
                if self.flip_remaining > 0 or self.final_phase == 'converge_to_precision':
                    self.previous_phase = self.current_phase
                    self.current_phase = 'precision_phase'
                else:
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                print("change current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
                return True

        if self.current_phase == 'precision_phase':
            if (current_precision > self.flip_precision
                and current_precision / (current_recall + 1e-8) > self.base_relative) \
                    or current_recall < self.base_recall or current_precision > self.max_performance_precision:
                if self.flip_remaining > 0:
                    self.previous_phase = self.current_phase
                    self.current_phase = 'recall_phase'
                    self.flip_remaining -= 1
                    print("changing current_phase to:", self.current_phase, 'flip_remaining', self.flip_remaining)
                    return True
                else:
                    assert self.final_phase == 'converge_to_precision'
                    self.previous_phase = self.current_phase
                    self.current_phase = self.final_phase
                    print("change current_phase to:", self.current_phase)
                    return True
        return False

    def show_status(self, current_recall=None, current_precision=None):
        print("epoch passed:", self.epoch_passed, "current phase:", self.current_phase,
              "relative_false_positive_penalty", self.relative_false_positive_penalty,
              "flip remaining:", self.flip_remaining)
        if current_recall is not None and current_precision is not None:
            print("current (recall, precision)", (current_recall, current_precision))

    def _update_relative_false_positive_penalty(self, current_recall, current_precision):

        if self.current_phase == 'warm_up':
            print("warm_up phase, relative_false_positive_penalty:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'recall_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.15
            print("recall phase, decrease relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'precision_phase':
            self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.13
            print("precision phase, increase relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_recall':

            if current_recall > self.target_recall:  # the recall is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty

        if self.current_phase == 'converge_to_precision':

            if current_precision > self.target_precision:  # the precision is higher than expected
                self.relative_false_positive_penalty = self.relative_false_positive_penalty / 1.025
                self.precision_to_recall_during_converging -= 1
                if self.precision_to_recall_during_converging <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_recall, current_precision)
                    exit()
            else:
                self.relative_false_positive_penalty = self.relative_false_positive_penalty * 1.024

            print("converging phase, change relative_false_positive_penalty to:", self.relative_false_positive_penalty)
            return self.relative_false_positive_penalty


if __name__ == '__main__':
    exit()
