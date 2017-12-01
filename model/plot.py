import sqlite3
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt 

class Ploter(object):
    '''
    plot some hist to view datas
    '''
    def __init__(self):
        self.load_data_record()
        self.load_data_time_interval()

    
    def load_data_time_interval(self,file_name="../data/time_interval"):
        pkl_file = open(file_name, 'rb')
        self.record_time_interval = pickle.load(pkl_file)
        self.record_session_length = pickle.load(pkl_file)
        self.record_usr_session = pickle.load(pkl_file)
        self.action_dict = pickle.load(pkl_file)
        pkl_file.close()

    def load_data_record(self,file_name="../data/record"):
        pkl_file = open(file_name, 'rb')
        player_record = pickle.load(pkl_file)
        pkl_file.close()
        self.labels = []
        for user_id in player_record:
            self.labels.append(player_record[user_id]['label'])  

    def plot_time_interval(self):
        plt.hist(self.record_time_interval, bins=30)
        plt.show()
    def plot_session_length(self):
        plt.hist(self.record_session_length, bins=30)
        plt.show()
    def plot_usr_session(self):
        plt.hist(self.record_usr_session, bins=30)
        plt.show()
    def plot_total(self):
        plt.subplot(2,2,1)
        plt.hist(self.record_time_interval, bins=30,range=(0,100))
        plt.subplot(2,2,2)
        plt.hist(self.record_session_length, bins=30,range=(0,500))
        plt.subplot(2,2,3)
        plt.hist(self.record_usr_session, bins=30,range=(0,500))
        plt.subplot(2,2,4)
        plt.hist(self.labels)
        plt.show()
    

if __name__ == "__main__":
    ploter = Ploter()
    #ploter.plot_time_interval()
    #ploter.plot_session_length()
    #ploter.plot_usr_session()
    ploter.plot_total()