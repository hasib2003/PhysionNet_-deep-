"""
Applies preprocessing techinques to the data as required
tmin and tmax are to be set from env

"""
import mne
import numpy as np
# import config_local
import config
import os 


class PreProcessor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def keep_specific_channels(raw, channels_to_keep):
        """
        Keep only the specified channels in the raw EEG data and drop the rest.

        Parameters:
        raw (mne.io.Raw): The raw EEG object.
        channels_to_keep (list of str): List of channel names to keep.

        Returns:
        mne.io.Raw: The modified raw EEG object with only the specified channels.
        """
        # Get the list of all channels in the raw object
        all_channels = raw.ch_names
        
        # Determine which channels to drop by finding the difference
        channels_to_drop = list(set(all_channels) - set(channels_to_keep))
        
        # Drop the channels
        raw.drop_channels(channels_to_drop)        
        return raw


    @staticmethod
    def pre_process(edf_raw):
        """
        takes input the raw object of mne performs pre_processing such as filtering
        return an other raw object
        """

        chns = ['C3..', 'C4..', "P3..", "P4..", "O1..", "O2.."]

        edf_reduced_channels = PreProcessor.keep_specific_channels(edf_raw,chns)



        return edf_reduced_channels
    
    # @staticmethod
    # def post_process(epoched_np):
    #     """ 
    #     performs post processing tasks such as features extraction on the epoched data from mne
    #     """

    #     # extracting the PSD features 
    #     Utlis.get_psd(epoched_np,)

    #     # return m




                
    @staticmethod
    def get_epochs(tarFiles,inlcude_rest=True):
        """
        filesNames represents a list of edf files
        returns the numpy array containing epoched data from given files
        Args:
            include_rest (boolean): determines weather to include the resting state data in between the trails (default:True)
        
        """

        subj_data = None

        # print("tarFiles ",tarFiles)


        for idx,file in enumerate(tarFiles):

            

            edf_data = mne.io.read_raw_edf(file, verbose=False,preload=True)


            # necessary processing steps
            # print("edf_data.ch_names ",edf_data.ch_names)
            # edf_data = PreProcessor.pre_process(edf_data)
            # print("edf_data.ch_names ",edf_data.ch_names)

            # edf_data = edf_data.filter(1,80)


            events, event_id = mne.events_from_annotations(edf_data)

            tmin = float(os.getenv("EPOCH_MIN"))  # start of each epoch relative to the event
            tmax = float(os.getenv("EPOCH_MAX"))   # end of each epoch relative to the event



            if inlcude_rest == False:
                event_id.pop('T0')

            # Create epochs from events
            epochs = mne.Epochs(edf_data, events, event_id, tmin, tmax,baseline=None,preload=True,verbose=False)        


            epoch_data = epochs.get_data(copy=True)

            # print("epoch_data.shap/e -> ",epoch_data.shape)


            if(subj_data is not None):
                subj_data = np.concatenate((subj_data,epoch_data),axis=0)
            else:
                subj_data = epoch_data

        return subj_data
    


if __name__ == "__main__":
    pass