#!/usr/bin/python
import math
import random
import os
import MySQLdb
import sys
sys.path.insert(0, '/home/rishabh/CoreInfrastructure/')
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as sk
from matplotlib import pyplot
#import matplotlib.pyplot as plt
import operator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class Trust_script:

    #trust_table = "small_result"
    #network_table = "small_network"
    #trust_table = "result_35"
    #network_table = "twitter_network_35"
    database =  "tt2"
    host = 'localhost'
    user = 'rishabh'
    password = ''
    cursor = None


    msg_tables = ["twt_msg_1stPart", "twt_msg_2ndPart",  "twitter_messagesEn34"]
    #msg_tables = ["twt_msg_1stPart", "twt_msg_2ndPart", "twt_msg_1stPart"]
    #topics = "cat_fb22_all_500t_cp_w"
    topics = "cat_met_a30_2000_cp_w"
    #feat_tables = [ "feat$"+topics+"$"+ msg_table +"$user_id$16to16" for msg_table in  msg_tables ]
    feat_tables = [ "feat$1to3gram$" + msg_table +"$user_id$16to16$0_01"  for msg_table in  msg_tables ]
    num_of_topics = 44777
    all_users_indices = {}
    time_user_topic  = np.empty((0))



    network_table = 'twitter_network_1001'
    topics = "cat_fb22_all_500t_cp_w"
    topics = 'cat_met_a30_2000_cp_w'
    msg_tables = ["msg_1001_nrt_core_1",  "msg_1001_nrt_core_2" ,  'msg_1001_nrt']
    msg_tables = ["msg_1001_nrt_core_1",  "msg_1001_nrt_core_2" ,  'msg_1001_nrt_1st']
    feat_tables = [ "feat$"+topics+"$"+ msg_table +"$user_id$16to16" for msg_table in  msg_tables ]
    num_of_topics = 2001
    # accounts_table = ['twitter_accounts_1001', 'twitter_accounts_1001_core']
    account_table = 'twitter_accounts_1001_uniques'

    word_to_int_dict = {}
    int_to_word_dict = {}


    def main(self):
        all_users_list = self.retrieve_users(self.account_table, 0, 200)
        core_users_list = self.retrieve_users(self.account_table, 1, 200)
        print ('2hops & 3 hops lengths: ' , len(core_users_list) , ' , ', len(all_users_list))

        # #### dictionary of user_id to int index
        # self.all_users_indices = { all_users_list[i] : i for i in range(len(all_users_list)) }
        # print ('(len(self.all_users_indices): ', len(self.all_users_indices ))

        #### dictionary of user_index to int index
        self.all_users_indices = { core_users_list[i] : i for i in range(len(core_users_list)) }
        print ('(len(self.all_users_indices): ', len(self.all_users_indices ))
        for uid in  all_users_list:
                if uid not in core_users_list:
                        self.all_users_indices[uid] = len(self.all_users_indices)
        print ('(len(self.all_users_indices): ', len(self.all_users_indices ))

        #### retrive the network between core_users to all_users: network: dict{ user_id: [user_ids] }
        network , all_users_list = self.retrieve_network(core_users_list)

        #### building random friends list for each user in core_users_list random_network: dict{ user_id: [user_ids] }
        random_network = self.random_draw( network, core_users_list, all_users_list)

        #### initializing 3 dimension array which keeps corresponding value for each feature table, and each user, and each topic/word.
        self.time_user_topic = np.zeros( ( len(self.msg_tables),len(self.all_users_indices) , self.num_of_topics) )
        print ( 'self.time_user_topic.shape : ', self.time_user_topic.shape )

        #### retrieve topics/words  by filling self.time_user_topic. As well as getting a list of words that should be later dropped from calculation.
        delete_list = self.retrieve_topics(core_users_list, all_users_list)


        #### calculate average use of each word, among friends. For both actual friends list and random friends list.
        friends_topics_avg, random_topics_avg  = self.friends_lang_avg(network, core_users_list)

        #### build delta_tt: language at t1 - language at t0 , delta_tf: language of friends - language at t0 , delta_tr: language of random friends - language at t0
        delta_tt, delta_tf, delta_tr = self.calculate_deltas(network, core_users_list, friends_topics_avg, random_topics_avg)
        #correlations = self.correlation( delta_tt, delta_tf)

        #### calculate the cosine similarity measure for both friends and random_friends.
        params, friends_param, order = self.learn_params(delta_tt, delta_tf, delta_tr, delete_list)
        #self.write_to_file(correlations, 'correlations.csv')


        print ('len(params): ' , len(params) )
        print ('len(params[0]) : ' , len(params[0]))

        friends_param = dict(zip( [self.int_to_word_dict[val] for  val in order], friends_param) )
        params = dict(zip( [self.int_to_word_dict[val] for  val in order], params ) )
        # cos_sims_r = dict(zip( [self.int_to_word_dict[val] for  val in order], cos_sims_r ) )

        print (params[:10])

        self.write_to_file(params, 'results/params.csv')
        self.write_to_file(friends_param, 'results/friends_param.csv')
        self.write_to_file(friends_param, 'results/friends_param_025.csv', 0.25)
        self.write_to_file(friends_param, 'results/friends_param_05.csv', 0.5)






    def connectMysqlDB(self):
        conn = MySQLdb.connect(self.host, self.user, self.password, self.database)
        c = conn.cursor()
        return c

    def retrieve_users(self, table, user_level, statuses_threshold):
        print ('db:retrieve_users' )
        users = []
        try:
            self.cursor = self.connectMysqlDB()
        except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
        if(self.cursor is not None):
            sql = "select distinct id from {0} where level <= {1} and statuses_count > {2}".format(table, user_level, statuses_threshold)
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            for row in result:
                users.append(row[0])
        print ('len(data):  '  , len(users) , '  ' ,  users[0] )
        return users


    def write_to_file(self, data, file_name, thresh = 0):
        f = open(file_name,'w')
        if isinstance(data , list):
            for elem in data:
                f.write(str( str(elem) + '\n'))
        elif isinstance(data, dict):
            for key,val in data.items():
                if isinstance(val, list) or type(val).__module__ == np.__name__ :
                    for elem in data:
                        f.write(str( str(elem) + '\t'))
                    f.write('\n')
                elif abs(val) > thresh:
                    f.write(str( str(key) + '\t,\t' + str(val) + '\n'))
        f.close()

    def learn_params(self, delta_tt, delta_tf, delta_tr, delete_list):
        print ('learn_params:' )
        print ( delta_tt.shape, ' , ', delta_tf.shape, ' , ', delta_tr.shape )
        params, friends_param, order = [], [], []
        for i in range(0,delta_tt.shape[1]):
            if i in delete_list:
                continue
            X = np.concatenate(( delta_tf[:,i].reshape(delta_tf.shape[0],1), delta_tr[:,i].reshape(delta_tr.shape[0],1) ), axis = 1)
            # X = np.concatenate(( X, np.ones(( delta_tf.shape[0], 1)) ), axis = 1)
            Y = delta_tt[:,i]



            reg = LinearRegression().fit(X, Y)

            Ypred = reg.predict(X)
            mae = mean_absolute_error(Y, Ypred)
            # coefs = np.concatenate( (reg.coef_, np.array([reg.intercept_, mae]) ) )
            coefs =  [ reg.coef_[0] , reg.coef_[1], reg.intercept_, mae ]


            if i% 1000 == 0:
                print ( 'i: ', i, ' , ',  X.shape, ' , ', Y.shape , ' , ' , len(coefs))
                print (len(params) , ' , ' , len(params[0]), ' , ', len(friends_param))
                print ('coefs: ', coefs)
                print (reg.coef_)
                print (params[:10])
            params.append( coefs )
            friends_param.append(reg.coef_[0])
            order.append(i)
        print ('len(params): ' , len(params) )
        return params , friends_param, order

    def cosine_sim(self, delta_tt, delta_tf, delta_tr, delete_list):
        print ('cosine_sim:' )
        print ( delta_tt.shape, ' , ', delta_tf.shape, ' , ', delta_tr.shape )
        cos_sim, cos_sim_r, order = [], [], []
        for i in range(0,delta_tt.shape[1]):
            if i in delete_list:
                continue
            data = np.concatenate((delta_tt[:,i].reshape(delta_tt.shape[0],1), delta_tf[:,i].reshape(delta_tf.shape[0],1)), axis = 1)
            data_r = np.concatenate((delta_tt[:,i].reshape(delta_tt.shape[0],1), delta_tr[:,i].reshape(delta_tr.shape[0],1)), axis = 1)
            if data.shape[0] < 1:
                continue
            if i% 1000 == 0:
                print ( 'i: ', i, ' , ',  data.shape, ' , ', data_r.shape )
            cos_sim.append(sk.cosine_similarity(np.transpose(data))[0][1] )
            cos_sim_r.append(sk.cosine_similarity(np.transpose(data_r))[0][1] )
            order.append(i)
        print ('len(cos_sim): ' , len(cos_sim) )
        return cos_sim , cos_sim_r, order


    def friends_lang_avg(self, network , core_users_list):
        print ( "calculate_friends_avg" )
        print ( 'len(network): ', len(network) )
        friends_topics_avg = np.zeros( ( len(core_users_list), self.num_of_topics) )
        random_topics_avg = np.zeros( ( len(core_users_list), self.num_of_topics) )
        print ( "len(friends_topics_avg): " , len(friends_topics_avg), ' , ', len(core_users_list) )

        for user_id in core_users_list:
            if user_id not in network:
                continue
            user_index = self.all_users_indices[user_id]
            friends = network[user_id]
            friends_indices = [ self.all_users_indices[fid] for fid in friends]
            friends_topics_avg[user_index ] = np.mean(self.time_user_topic[2, friends_indices, :], axis = 0)
        print ('friends_topics_avg.shape: ', friends_topics_avg.shape)

        average = np.mean(self.time_user_topic[2 , : , : ] , axis = 0)
        for user_id in core_users_list:
            if user_id not in network:
                continue
            user_index = self.all_users_indices[user_id]
            random_topics_avg [user_index ] = average
        print ('random_topics_avg.shape: ', random_topics_avg.shape)

        return friends_topics_avg, random_topics_avg



    def random_draw(self, network, core_users_list, all_users_list):
        print ( 'random_draw' )
        random_network = { k : [] for k in core_users_list }
        for user_id in core_users_list:
            # friends_count = len(network[user_id])
            # random_indices = random.sample(range(len(all_users_list)), k = friends_count)
            # random_network[user_id] = [ all_users_list[i] for i in random_indices ]
            random_network[user_id] = all_users_list
        return  random_network



    def retrieve_network(self, core_users_list):
        print ("retrieve_network" )
        network = {}

        updated_all_users_list = set(core_users_list)
        print ('len(updated_all_users_list): ', len(updated_all_users_list) , ' len(core_users_list): ', len(core_users_list))
        try:
            self.cursor = self.connectMysqlDB()
        except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
        if(self.cursor is not None):
            ####  save network as adjacency lists. A dictionary of nodes, in which For each node we keep its neighbors in a list
            network = { k : [] for k in core_users_list }
            #### for each node in two hops distance, we fetch its neighbors from db.
            for user_id in core_users_list:
                sql = "select * from {0} where source_id={1}".format(self.network_table, user_id)
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                friends = []
                for row in result:
                    #### only add the neighbors, for which their language info is available
                    if  row[2] in self.all_users_indices:
                        friends.append(row[2])
                # friends  = random.sample(friends, min(len(friends), 100) )
                network[user_id] = friends
                updated_all_users_list = updated_all_users_list.union(set(friends))
        print ('len(updated_all_users_list): ', len(updated_all_users_list) )
        return network, list(updated_all_users_list)



    def retrieve_topics(self, core_users_list, all_users_list):
        print ("retrieve_topics" )
        try:
            self.cursor = self.connectMysqlDB()
        except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
        if(self.cursor is not None):
            words =[]
            #### for loop over all three feature tables ( 1st part, 2nd part, total )
            for i in range(len(self.feat_tables)):
                #### extract list of all distinct words.
                sql = "select distinct (feat) from {0}".format(self.feat_tables[i])
                print ('sql: ', sql )
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                for row in result:
                    if not row[0] in words:
                        words.append(row[0])
                print ('len(words): ' , len(words) )


            #### make dictionaries of word to int and the inverse
            #word_to_int_dict = {}
            #int_to_word_dict = {}
            counter = 1
            for word in words:
                self.word_to_int_dict[word] = counter
                self.int_to_word_dict[counter] = word
                counter +=1
            print ('counter: ' , counter)


            for i in range(len(self.feat_tables)):
                print ('i: ' , i)
                sql = "select * from {0} where group_id in ({1})".format(self.feat_tables[i], ','.join([str(k) for k in all_users_list]) )
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                counter = 0
                print ('len(result): ', len(result))
                for row in result:
                    counter+=1
                    if counter % 1000000 == 0:
                        print ( 'counter: ' , counter )
                    #### drop rows that users are not in all_users_indices list
                    if ( row[1] not  in self.all_users_indices ):
                        continue
                    #### for feature tables 0 and 1, drop those rows that users are not in two hops distance
                    if ( i < 2) and  ( row[1] not in core_users_list):
                        continue
                    #### fill time_user_topic array. where by 'topic'/'word'
                    user = self.all_users_indices[row[1]]
                    topic = self.word_to_int_dict[row[2]]
                    #print ('i: ', i , ', user: ', user, ' topic: ', topic)
                    self.time_user_topic[i, user, topic] = row[4]

            #### since we want to normalize the data, we need the variance not to be zero,
            #### so we drop those words having 0 variance, by adding them to the delete_list
            delete_list = []
            variance = np.var(self.time_user_topic,1)
            average = np.mean(self.time_user_topic,1)
            for k in range(self.time_user_topic.shape[2]):
                if k in delete_list:
                    continue
                for i in range(self.time_user_topic.shape[0]):
                    if variance[i,k] == 0:
                        delete_list.append(k)
                        break
            print ('delete_list.size: ' , len(delete_list) )
            #print 'delete_list:'
            #for i in delete_list:
            #        print 'delete_list, i:', i ,  ' ,  '



            #### normalize data over all users, for each word.
            for k in range(self.time_user_topic.shape[2]):
                if k in delete_list:
                    continue
                for i in range(self.time_user_topic.shape[0]):
                    self.time_user_topic[i,:,k] =   (self.time_user_topic[i,:,k] - average[2,k]) / variance[2,k]
            print (' len(delete_list): ' , len(delete_list) )
            return delete_list


    def calculate_deltas(self, network, core_users_list, friends_topics_avg, random_topics_avg):
        print ("calculate_deltas" )
        delete_mask = []
        delta_tt = np.zeros(( len(core_users_list), self.num_of_topics))
        delta_tf = np.zeros(( len(core_users_list), self.num_of_topics))
        delta_tr = np.zeros(( len(core_users_list), self.num_of_topics))
        for user_id in core_users_list:
            user_index = self.all_users_indices[user_id]
            delta_tt[user_index] = np.subtract(self.time_user_topic[1,user_index,:] , self.time_user_topic[0,user_index,:] )
            delta_tf[user_index] = np.subtract(friends_topics_avg[user_index] , self.time_user_topic[0,user_index,:] )
            delta_tr[user_index] = np.subtract(random_topics_avg[user_index] , self.time_user_topic[0,user_index, :] )
            if len(network[user_id]) < 10:
                    delete_mask.append(user_index)
        print ('shapes: ' )
        print (delta_tt.shape )
        print (delta_tf.shape )
        print (delta_tr.shape )

        delta_tt = np.delete(delta_tt, delete_mask, 0)
        delta_tf = np.delete(delta_tf, delete_mask, 0)
        delta_tr = np.delete(delta_tr, delete_mask, 0)
        print ("len(delete_mask): " , len(delete_mask) )

        print ("self.delta_tt.shape: ", delta_tt.shape )
        print ("self.delta_tf.shape: ", delta_tf.shape )
        print ("self.delta_tr.shape: ", delta_tr.shape )
        return delta_tt, delta_tf, delta_tr


ts = Trust_script()
ts.main()