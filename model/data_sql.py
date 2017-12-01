import sqlite3
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from gensim.models import word2vec 
import re
class Parser(object):
    '''
    download data from db and process it.
    Transform the data format into a dict :

    key:  
        usr: user_id
    values:
        sessions:[session_one,session_two,...], session_one = [activity_one,activity_two,...]
        label: 0 or 1 means loss or stay player
    '''
    def __init__(self, sql_in="../data/kbzy.db"):
        self.sql_in = sql_in
        self.regx = re.compile(r'board_layer/board.*?')
        self.total_sessions = []
        self.embedding_size = 25
    def sql_data_from_db(self, interval_split_threshold=10):
        '''
        sql data from db, just select the data from where currday = 1
        Args: 
            interval_split_threshold: divide actions into different sessions by the timestamp of actions
        Returns:
            record_time_interval:  the time interval of all actions to plot and decide the
            interval split threshold 
        '''
        conn = sqlite3.connect(self.sql_in)
        c = conn.cursor()
        query_sql = "SELECT user_id, action, relative_timestamp, num_days_played \
            FROM maidian WHERE current_day = 1 ORDER BY user_id, relative_timestamp"
        
        result = {}
        curr_usr = -1
        curr_sessions = []
        sub_sessions = []
        record_time_interval = []
        record_session_length = []
        record_usr_session = []
        action_dict = {}
        action_index = 1
        user_num = 0
        for row in c.execute(query_sql):
            user_id = row[0]
            if (user_id != curr_usr):
                if (curr_usr != -1):
                    user_num += 1
                    print("user have changed {}".format(user_num))
                    curr_sessions.append(sub_sessions)
                    self.total_sessions.append(sub_sessions)
                    record_session_length.append(len(sub_sessions))
                    sub_sessions = []
                    result[curr_usr] = {}
                    result[curr_usr]['sessions'] = curr_sessions
                    record_usr_session.append(len(curr_sessions))
                    result[curr_usr]['label'] = curr_label
                curr_usr = user_id
                curr_sessions = []
                if row[3] >=2 :
                    curr_label = 1
                elif row[3] == 1:
                    curr_label = 0
                sub_sessions = []
                pre_timestamp = row[2]
                action = 'substitite_for_chat' if self.regx.match( \
                       row[1]) else row[1]
                    
                if action not in action_dict:
                    action_dict[action] = action_index
                    action_index += 1
                sub_sessions.append(action_dict[action])
            else:
                curr_timestamp = row[2]
                time_interval = curr_timestamp - pre_timestamp
                record_time_interval.append(time_interval)
                pre_timestamp = curr_timestamp
                if time_interval > interval_split_threshold:
                    curr_sessions.append(sub_sessions)
                    self.total_sessions.append(sub_sessions)
                    record_session_length.append(len(sub_sessions))
                    sub_sessions = []
                    action = 'substitite_for_chat' if self.regx.match( \
                       row[1]) else row[1]
                    if action not in action_dict:
                        action_dict[action] = action_index
                        action_index += 1
                    sub_sessions.append(action_dict[action])
                else:
                    action = 'substitite_for_chat' if self.regx.match( \
                       row[1]) else row[1]
                    if action not in action_dict:
                        action_dict[action] = action_index
                        action_index += 1
                    sub_sessions.append(action_dict[action])
        curr_sessions.append(sub_sessions)
        self.total_sessions.append(sub_sessions)
        record_session_length.append(len(sub_sessions))
        result[curr_usr] = {}
        result[curr_usr]['sessions'] = curr_sessions
        record_usr_session.append(len(curr_sessions))
        result[curr_usr]['label'] = curr_label
        with open("../data/record","wb") as f_out:
            pickle.dump(result, f_out)
        #free(result)
        with open("../data/time_interval","wb") as f_out:
            pickle.dump(record_time_interval, f_out)
            pickle.dump(record_session_length, f_out)
            pickle.dump(record_usr_session, f_out)
            pickle.dump(action_dict, f_out)
        print(action_index)
        return record_time_interval

    def word2vec_training(self, file_out):
        sessions = []
        for session in self.total_sessions:
            sessions.append([str(action) for action in session])
        self.model = word2vec.Word2Vec(
            sessions, self.embedding_size, min_count=1)
        self.model.save(file_out)
        


if __name__ == "__main__":
    parse = Parser()
    record_time_interval = np.asarray(parse.sql_data_from_db())
    parse.word2vec_training("../data/activity_embedding")

        
        


        


        


            


        



