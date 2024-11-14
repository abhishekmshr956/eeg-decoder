import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
# sys and os needed to make imports from parent directory
import sys
import os
# sys.path.append("/home/brandon/eeg/bci_raspy")
# import data.data_util as util
import utils

class TaskData():
    '''
    TaskData is a collection of all the data logged during a task 
    in the BCI-Raspy system along with a set of methods to perform 
    operations on that data.

    Things like the EEG signal, the EEGNet SoftMax predictions, the 
    decoder positions, and the trial types are all stored within 
    the task data
    '''
    def __init__(self, data_dir):
        '''
        Initialize a task data object by passing it the path of the 
        folder containing 'task.bin' and 'eeg.bin'
        

        data_dir (str): path to the directory containing the data
        '''
        self.data_dir = data_dir

        # load in the data
        self.eeg_bin = util.load_data(self.data_dir + "/eeg.bin")
        self.task_bin = util.load_data(self.data_dir + "/task.bin")

        # lists EEG data by trial-type
        self.eeg_trials = defaultdict(lambda: [])    
        # lists SoftMax data by trial-type
        self.softmax_probs = defaultdict(lambda: []) 

    def get_softmax_probs(self, trial_type=None, all_trials=None):
        '''
        Returns the SoftMax probabilities predicted by EEGNet. By 
        default this will return all the SoftMax probabilities 
        predicted by EEGNet over the duration of the entire experiment.
        Setting flag arguments will result in only returning SoftMax 
        probabilities during a part of the expirement.

        Arguments:
        trial_type (int): default is None, if specified only SoftMax probabilities 
            predicted during that trial type are returned. Trial type should 
            be specified as an int.
        all_trials (str): default is None. 
                -'correct' returns only the SoftMax probabilities for which the 
                    decoder hit the correct target. 
                -'miss' returns only the SoftMax probabilities for trials were 
                    the decoder did not hit the target. 
                -'wrong' returns only the SoftMax probabilities for trials were 
                    the decoder hit the incorrect target

        Returns:
        probs: the softmax probabilities having shape (n_trials, tsteps, 4)
        '''
        if trial_type != None:
            # list of (t_steps, 4) softmax probs
            probs = self.softmax_probs[int(trial_type)]
            probs = np.array(probs)  # (n_trials, tsteps, 4)
            if all_trials=='correct':
                probs = probs[self.correct_trial_flags[int(trial_type)]==True]
            if all_trials=='miss':
                probs = probs[self.correct_trial_flags[int(trial_type)]==False]
        else:
            tmp = []
            tmp.append(np.array(self.softmax_probs[0]))
            probs = np.vstack(tmp)
        return probs

    def partition_trials(self):
        '''partitions all the data into trials'''\

        #-------------------------------------------------------#
        # EXTRACT TRIAL START TIMES AND INDICES
        #-------------------------------------------------------#
        new_state_bool = np.zeros(self.task_bin['state_task'].size, dtype='bool')
        new_state_bool[1:] = (self.task_bin['state_task'][1:] != self.task_bin['state_task'][:-1])
        self.trial_start_ixs = np.nonzero(new_state_bool)[0]
        self.trial_labels = self.task_bin['state_task'][self.trial_start_ixs]
        self.trial_start_times = self.task_bin['time_ns'][self.trial_start_ixs] 
        self.eeg_timing = self.eeg_bin['time_ns']
        


        #------------------------------------------------------#
        # PARTITION EEG DATA TO TRIALS
        #------------------------------------------------------#
        n_trials = len(self.trial_labels)
        for trial_ix in range(n_trials-1):
            if int(self.trial_labels[trial_ix]) == 4:  # skip rest trials
                continue 

            # extract eeg_data indices were trial started and ended
            trial_start_ix = np.where(self.task_bin["time_ns"] > self.trial_start_times[trial_ix])[0][0] + 50 # omit first second of data
            trial_end_ix = np.where(self.task_bin["time_ns"] < self.trial_start_times[trial_ix+1])[0][-1]
            
            # get the decoder position
            self.eeg_trials[self.trial_labels[trial_ix]].append(self.task_bin["decoded_pos"][trial_end_ix])

            # get SoftMax predictions for this trial
            self.softmax_probs[self.trial_labels[trial_ix]].append(self.task_bin['decoder_output'][self.trial_start_ixs[trial_ix]:self.trial_start_ixs[trial_ix+1]])

    def compute_accuracy(self):
        '''computes the accuracy of the decoder for all trials'''
        # maps target label to decoder 2D (x,y) position
        target2pos = {
            0 : [-75, 0],  # left
            1 : [75, 0],   # right
            2 : [0, -75],  # down
            3 : [0, 75]    # up
        }

        # how close decoder must be to target to be correct
        target_thresh = 30   # 2/3 of the way to target

        # maps each target to a list of bools indicating if that trial is correct
        correct_trials = defaultdict(lambda: 0)  
        self.correct_trial_flags = defaultdict(lambda: 0)
        n_trials = defaultdict(lambda: 0)  # number of trials for each condition

        # list of decoder positions at the end of all trials
        # e.g. final_decoder_pos[0] is a list of (x,y)-coordinates 
        # of the final decoder position at the end of each trial 
        # associated with left (target=0) reaches
        final_decoder_pos = defaultdict(lambda: 0)
        decoder_acc = defaultdict(lambda: 0)
        for target in range(4):
            # eeg_trials[target] is a list with length equal to the number of target trials.
            # Each array is a timeseries of 2D position of the decoder for that trial
            final_decoder_pos[target] = np.array(self.eeg_trials[target])  # final x,y-position of decoder on each trial (n_trials, 2) 
            
            # get distnace from target at the end of each trial
            distances = distance_from_target(final_decoder_pos[target], target2pos[target])
            # sum the number of trials that ended close to the target
            self.correct_trial_flags[target] = distances < target_thresh
            correct_trials[target] = np.sum(self.correct_trial_flags[target])
            
            # total number of this type of trial
            n_trials[target] = final_decoder_pos[target].shape[0] 
            
            # accuracy on this trial type
            acc = correct_trials[target]*100./n_trials[target]
            decoder_acc[target] = np.maximum(0., acc)  # negative accuracy indicates unknown class
            if True:
                print("\nTarget:", target)
                print("correct trials:", correct_trials[target])
                print("trials of this type:", n_trials[target])
                print("accuracy(%):", acc )

        total_correct_trials = 0
        total_trials = 0
        for target in range(4):
            total_correct_trials += correct_trials[target]
            total_trials += n_trials[target]
        acc = total_correct_trials * 100. / total_trials
        print("\n\nAverage accuracy across all trials(%):", acc)
            
        # decoder_acc[target_ix] = accuracy_of_decoder_for_that_target


#------------------------------------------------------------------------#
# HELPER FUNCTIONS
#------------------------------------------------------------------------#
def distance_from_target(final_decoder_pos, target_pos):
    '''returns distance of decoder from the target
    
    parameters
    -final_decoder_pos: NumPy array of the final decoder position in 2D. Has shape
        (n_trials, 2)
    -target_pos: NumPy array or list of the target position in 2D. If numpy array 
        has shape (2,) or if list has length 2
    '''
    
    x_dist_square = (final_decoder_pos[:,0] - target_pos[0])**2
    y_dist_square = (final_decoder_pos[:,1] - target_pos[1])**2
    
    dist = np.sqrt(x_dist_square + y_dist_square)

    return dist   # distance from target at the end of each trial (n_trials,)

def plot_softmax_distribution(probs, bins=10):
    '''
    plots the distribution of softmax probabilities


    Arguments
    probs: a numpy array (trials, tsteps, N-way-softmax)
    '''
    # histogram wants #(4, samples)
    n_classes = probs.shape[-1]
    probs = probs.reshape(-1, n_classes)
    probs = np.transpose(probs)

    distr = []
    for i in range(n_classes):
        distr.append(probs[i])

    # plot the histogram
    plt.hist(distr, bins=10)
