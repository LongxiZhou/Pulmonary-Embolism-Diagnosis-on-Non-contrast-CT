import numpy as np


class OutlierLossDetect:
    def __init__(self, store_count=30, remove_max_count=3, remove_min_count=3, std_outlier=7):
        self.recent_loss_history = []
        self.store_count = store_count
        self.remove_max_count = remove_max_count
        self.remove_min_count = remove_min_count
        self.std_outlier = std_outlier

        self.in_queue_count = 0

        self.consecutive_outlier = 0

    def update_new_loss(self, new_loss):
        # return True for non-outlier loss

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
    
    
class TrainingPhaseControlFlowRoughness:
    """
    flow roughness is quantified by the max Jacobi determinant

    roughness = max(jacobi_determinant)
    """
    def __init__(self, params):
        self.target_rough = params["target_rough"]  # e.g., 2
        self.flip_high_rough = params["flip_high_rough"]  # e.g., 5
        self.flip_low_rough = params["flip_low_rough"]  # e.g., 1.5

        self.current_phase = 'warm_up'
        # 'warm_up', 'increase_rough', 'decrease_rough', 'converge_to_target_rough'
        
        self.final_phase = 'converge_to_target_rough'

        self.flip_remaining = params["flip_remaining"]
        # one flip means change the phase 'precision_phase' -> 'recall_phase'

        self.cross_between_target_during_converge = 3

        self.relative_penalty_for_flow = params["relative_penalty_for_flow"]
        # higher means more penalty for un-smooth

        self.warm_up_epochs = params["warm_up_epochs"]

        self.previous_phase = None
        self.previous_roughness = None
        self.changed_phase_in_last_epoch = False
        
        self.history = []  # (roughness, current_phase, penalty_for_roughness)
        self.epoch_passed = 0
        
    def get_new_relative_penalty_for_flow(self, current_roughness):
        self._update_history(current_roughness)
        self.changed_phase_in_last_epoch = self._update_phase(current_roughness)
        self._update_penalty_for_roughness(current_roughness)
        self.show_status(current_roughness)
        self.epoch_passed += 1
        return self.relative_penalty_for_flow
        
    def _update_history(self, roughness):
        self.history.append((roughness, self.current_phase, self.relative_penalty_for_flow))
    
    def _update_phase(self, current_roughness):
        # return True for phase change

        self.previous_roughness = current_roughness
        self.previous_phase = self.current_phase

        if self.current_phase == 'warm_up' and self.epoch_passed >= self.warm_up_epochs:
            self.current_phase = 'decrease_rough'
            print("changing current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
            return True
        
        if self.current_phase == 'decrease_rough':
            if current_roughness < self.flip_low_rough:
                if self.flip_remaining > 0:
                    self.current_phase = 'increase_rough'
                    self.flip_remaining -= 1
                    print("changing current_phase to:", self.current_phase, 'flip_remaining', self.flip_remaining)
                    return True
                else:
                    assert self.final_phase == 'converge_to_target_rough'
                    self.current_phase = self.final_phase
                    print("change current_phase to:", self.current_phase)
                    return True

        if self.current_phase == 'increase_rough':
            if current_roughness > self.flip_high_rough:
                self.current_phase = 'decrease_rough'
                print("changing current_phase to:", self.current_phase, "previous phase:", self.previous_phase)
                return True

        return False

    def show_status(self, current_roughness=None):
        if current_roughness is not None:
            print("current_roughness:", current_roughness)
        print("epoch passed:", self.epoch_passed,
              "current phase:", self.current_phase,
              "penalty_for_roughness", self.relative_penalty_for_flow,
              "flip remaining:", self.flip_remaining)

    def _update_penalty_for_roughness(self, current_roughness):

        if self.current_phase == 'warm_up':
            print("warm_up phase, penalty for roughness:", self.relative_penalty_for_flow)
            return self.relative_penalty_for_flow

        if self.current_phase == 'increase_rough':
            self.relative_penalty_for_flow = self.relative_penalty_for_flow / 1.15
            print("increase_rough phase, decrease penalty for roughness to:", self.relative_penalty_for_flow)
            return self.relative_penalty_for_flow

        if self.current_phase == 'decrease_rough':
            self.relative_penalty_for_flow = self.relative_penalty_for_flow * 1.13
            print("decrease_rough phase, increase penalty for roughness to:", self.relative_penalty_for_flow)
            return self.relative_penalty_for_flow

        if self.current_phase == 'converge_to_target_rough':
            if current_roughness > self.target_rough:  # the rough is higher than expected
                self.relative_penalty_for_flow = self.relative_penalty_for_flow * 1.024
                if self.previous_roughness < self.target_rough:
                    self.cross_between_target_during_converge -= 1
                if self.cross_between_target_during_converge <= 0:
                    print("Training Finished, final status:")
                    self.show_status(current_roughness)
                    exit()
            else:
                self.cross_between_target_during_converge = self.cross_between_target_during_converge / 1.025

            print("converging phase, change relative_penalty_for_roughness to:", self.relative_penalty_for_flow)
            return self.relative_penalty_for_flow


if __name__ == '__main__':
    exit()
