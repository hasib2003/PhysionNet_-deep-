"""Loads dataset required by the model"""

# working with subsets of files for one subject is not supported yet

from torch.utils.data import Dataset
import os
import mne
import numpy as np

# custom
import config_colab
# import config
from pre_processors import PreProcessor
from utlis import Utlis


class PSD_PhysioNet(Dataset):
    def __init__(self,raw_dataset,freq_bin,duration) -> None:
        """
        Args:
            raw_dataset : the object of Raw_PhysioNet class
            freq_bin: dictionary containing frequency bins to be used
            duration: percentage of total signal to be used for psd extraction
        """

        super().__init__()
        self.raw_x = raw_dataset.eeg_data_x
        self.raw_y = raw_dataset.eeg_data_y.reshape(-1,)

        self.freq_bin = freq_bin 
        self.duration = duration

        self.eeg_psds = Utlis.get_psd(raw_eeg_signal=self.raw_x,duration=self.duration,bin_dict=self.freq_bin,use_simp=True)[0]
        # self.eeg_psds = np.transpose()



    def __len__(self):


        return len(self.eeg_psds)


    def __getitem__(self, index) :

    
        return self.eeg_psds[index] , self.raw_y[index]


class Raw_PhysionNet(Dataset):
    """ contains raw epoched data in form of np arrays """
    def __init__(self,activity="all") -> None:
        super().__init__()


        self.activity_name = activity
        self.activity_list = ["eye_open","eye_closed","fist_real","fist_imagined","both_real","both_imagined","all"]

        """
        Args:
            activity: determines which activity is to be used for authenticaion prupose
                        must be one of the following 
                            1. fist_real -> subject opens and closes the corresponding fist until the target disappears.
                            2. fist_imagined -> The subject imagines opening and closing the corresponding fist until the target disappears
                            3. both_real -> The subject opens and closes either both fists (if the target is on top) or both both (if the target is on the bottom) until the target disappears.
                            4. both_imagined -> The subject imagines opening and closing either both fists (if the target is on top) or both both (if the target is on the bottom) until the target disappears
                            5. eye_open -> baseline task
                            6. eye_closed -> baseline task
                            7. all
        """

        assert activity in self.activity_list, f"activity {activity} not in {self.activity_list}"

        self.sample_rate = 160
        self.window_length = 0.5
        self.window_size = int(self.sample_rate * self.window_length) # number of points to be used

        self.root_dir = os.getenv('PATH_ROOT_DIR')  # root dir containing all files
        self.path_records = os.getenv('PATH_RECORDS') # file containg names of all .edf files
        

        if self.activity_name != "all":
            self.files_per_subj =  3  # files per subject
        else:
            self.files_per_subj = 14






        self.num_subjs =  int(os.getenv("NUM_SUBJS"))  # total number of subjs

        assert self.files_per_subj <= 14, f"Each subject can have max 14 files"
        assert self.num_subjs <=109 , f"Max 109 subjects are allowed" 

  
        assert os.path.isdir(self.root_dir) , f"No directory found at {self.root_dir}"
        assert os.path.isfile(self.path_records), f"No file found at {self.path_records}"

        self.eeg_data_x = None
        self.eeg_data_y = None
        self.load_meta()
        self.load_mem()


        # self.eeg_data_x = np.transpose(self.eeg_data_x,(1,0))
    
    def filter_paths(self,file_name):
        """
        filters the paths according to the activity required by the user
            filtering assumes following structure
                file1 -> baseline, eyes open
                file2 -> baseline, eyes close
                file3 -> Task 1 (open and close left or right fist)
                file4 -> Task 2 (imagine opening and closing left or right fist)
                file5 -> Task 3 (open and close both fists or both both)
                file5 -> Task 4 (imagine opening and closing both fists or both both)
                .
                .
                .        
        """

        idx = self.activity_list.index(self.activity_name) + 1
        
        if idx == len(self.activity_list) : # if activity_name is "all"  always returning trye
            return True 


        file_num = int(file_name.split("R")[-1].split(".")[-2])
        if idx > 2: 
            # if task is not eyes open or close
            if file_num >= idx and (file_num - idx ) % 4 == 0:
                return True 
            return False

        else:
            return idx == file_num



    def load_meta(self):

        """
        searches for the paths of the edf files and load them into mem
        """

        file_names = []

        with open(self.path_records,"r") as f:
            
            content = f.readline()
            while(content):
                file_name = content.strip()
                # # checking if the file belongs to the given activity
                if self.filter_paths(file_name) == False:
                    content = f.readline()  
                    continue
                
                file_names.append(file_name)
                content = f.readline()


        # print(f"following records found {file_names}")

        assert len(file_names) >= self.num_subjs * self.files_per_subj , f"Expected {self.num_subjs * self.files_per_subj} or more files. Found {len(file_names)}"
        # allowing more files to be able to work with subsets of data as well

        self.files = [] # list containg the paths of all edf. files

        for file_name in file_names:
            self.files.append(os.path.join(self.root_dir,file_name))
        
        
        # print(f"files with path {self.files}")
        print(f".... found {len(self.files)} edf files ....")

    def load_mem(self):
        """
        loads data into the memory in form of np arrays
        """

        for id in range(self.num_subjs):

            subj_data = self.load_subj(id+1)        
            subj_ids = np.full((subj_data.shape[0],1),fill_value=id)

            if(self.eeg_data_x is not None):
                self.eeg_data_x = np.concatenate((self.eeg_data_x,subj_data),axis=0)
                self.eeg_data_y = np.concatenate((self.eeg_data_y,subj_ids),axis=0)

            else:
                self.eeg_data_x = subj_data
                self.eeg_data_y = subj_ids
        
    def load_subj(self,subj_id):
        """
        given the subject id i.e. 1,2,3 ...
        returns the all of epochs of that subject
        """

        start_idx = (subj_id - 1) * self.files_per_subj
        end_idx = start_idx + self.files_per_subj

        subj_files = self.files[start_idx:end_idx] # contains files related to concerned subject only         
        # print(f"subj with id {subj_id} files are ",subj_files)
        subj_data = PreProcessor.get_epochs(subj_files,inlcude_rest=True)
        # print("subj_data.shape ",subj_data.shape)
        

        return subj_data
    
    def standardize_rows(self,arr):
        """
        Standardize each row of the array to have a mean of 0 and a standard deviation of 1.
        
        Parameters:
        arr (numpy.ndarray): Input array of shape (64, 80)
        
        Returns:
        numpy.ndarray: Standardized array with the same shape as input
        """
        # Calculate the mean and standard deviation for each row
        row_mean = arr.mean(axis=1, keepdims=True)
        row_std = arr.std(axis=1, keepdims=True)
        
        # Standardize each row
        standardized_arr = (arr - row_mean) / row_std
        
        return standardized_arr

    def __len__(self):
        # because of the sliding window

        return self.eeg_data_x.shape[0]  * 4 


    def __getitem__(self, index) :

        # using sliding window approach with no overlapping to obtain a 0.5 sec window

        # total length of signal  = 2 with 321 points (160 * 2)

        sample_idx = index // 4
        window_idx = index % 4
        



        sample = self.eeg_data_x[sample_idx][:,window_idx*self.window_size:(window_idx+1)*self.window_size]
        sample = self.standardize_rows(sample) # z score normalization

        return sample , self.eeg_data_y[sample_idx][0]



if __name__ == "__main__":
    data = Raw_PhysionNet(activity="all")
    print(f"--total samples {data.__len__()} --")
    for i in data:
        print("x shape ->",i[0].shape)
        print("y-> ",i[1])
        break
        # input()