"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def batch_convert(self, batch):
        combined_batch = {}
        for batch_dict in batch:
            for key, value in batch_dict.items():
                if key not in combined_batch:
                    combined_batch[key] = []
                combined_batch[key].append(value)

        batch_dict_array = {}
        for key, value in combined_batch.items():
            batch_dict_array[key] = np.array(value)

        return batch_dict_array

    def __iter__(self):
        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        if self.shuffle is True:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = range(0, len(self.dataset))
            
        batch = []
        
        for i in indices:
            batch.append(self.dataset[i])
            if len(batch) == self.batch_size:
                yield self.batch_convert(batch)
                batch = []

        if self.drop_last is False and len(batch) > 0:
            yield self.batch_convert(batch)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last!                                 #
        ########################################################################

        if self.drop_last is True:
            length = len(self.dataset) // self.batch_size
        else:
            length = len(self.dataset) // self.batch_size + 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
