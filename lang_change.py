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
    accounts_table = ['twitter_accounts_1001', 'twitter_accounts_1001_core']

    word_to_int_dict = {}
    int_to_word_dict = {}


    def main(self):
        three_hops_users_list = self.retrieve_users(self.accounts_table[0], 50)
        two_hops_users_list = self.retrieve_users(self.accounts_table[1], 100)
        #two_hops_users_list = two_hops_users_list[:100]
        #print trust_df
        #print trust_df.avg_friends
        #two_hops_users_list = trust_df[trust_df.num_of_friends>1].index.tolist()
        #three_hops_users_list = trust_df.index.tolist()
        print ('2hops & 3 hops lengths: ' , len(two_hops_users_list) , ' , ', len(three_hops_users_list))

        #### dictionary of user_index to int index
        self.all_users_indices = { two_hops_users_list[i] : i for i in range(len(two_hops_users_list)) }
        print (len(self.all_users_indices ))
        for uid in  three_hops_users_list:
                if uid not in two_hops_users_list:
                        self.all_users_indices[uid] = len(self.all_users_indices)
        print (len(self.all_users_indices ))


        network , updated_all_users_list = self.retrieve_network(two_hops_users_list, three_hops_users_list)

        #### filling trust_df dataframe, as well as building random friends list for each user in two hops distance by selecting randomly
        #### picking some friends from users in at most three hops distance to the root.
        random_friends = self.random_draw( network, three_hops_users_list, two_hops_users_list)

        #### initializing 3 dimension array which keeps corresponding value for each feature table, and each user, and each topic(here word).
        self.time_user_topic = np.zeros( ( len(self.msg_tables),len(self.all_users_indices) , self.num_of_topics) )
        print (len(self.time_user_topic) )
        print ( self.time_user_topic.shape )

        #### retrieve topics ( here words ) by filling self.time_user_topic. As well as getting a list of words that should be later dropped from calculation.
        delete_list = self.retrieve_topics(two_hops_users_list, updated_all_users_list)
        print ( 'retrieved topics' )

        #### calculate average use of each word, among friends. For both actual friends list and random friends list.
        friends_topics_avg, random_topics_avg  = self.friends_lang_avg(network, two_hops_users_list, random_friends)

        #### build delta_tt: language at t1 - language at t0 , delta_tf: language of friends - language at t0 , delta_tr: language of random friends - language at t0
        delta_tt, delta_tf, delta_tr = self.calculate_deltas(network, two_hops_users_list, friends_topics_avg, random_topics_avg)
        #correlations = self.correlation( delta_tt, delta_tf)

        #### calculate the cosine similarity measure for both friends and random_friends.
        cos_sims  , cos_sims_r, order = self.cosine_sim(delta_tt, delta_tf, delta_tr, delete_list)
        #self.write_to_file(correlations, 'correlations.csv')


        print ('len(cos_sim): ' , len(cos_sims), 'cos_sim_mean: ' , np.mean(cos_sims) )
        print ('len(cos_sim_r): ' , len(cos_sims_r), 'cos_sim_r_mean: ' , np.mean(cos_sims_r) )

        cos_sims_dif = dict(zip( [self.int_to_word_dict[val] for  val in order], map(operator.sub,cos_sims , cos_sims_r) ) )
        cos_sims = dict(zip( [self.int_to_word_dict[val] for  val in order], cos_sims ) )
        cos_sims_r = dict(zip( [self.int_to_word_dict[val] for  val in order], cos_sims_r ) )

        #self.write_to_file(cos_sims, '1_cos_sims.csv')
        #self.write_to_file(cos_sims_r, '1_cos_sims_r.csv')
        self.write_to_file( cos_sims_dif, '1_cos_sims_dif.csv')
        self.write_to_file( cos_sims_dif, '1_cos_sims_dif_0.05.csv', 0.05)
        #self.write_to_file( cos_sims_dif, '1_cos_sims_dif_0.1.csv', 0.1)




    def connectMysqlDB(self):
        conn = MySQLdb.connect(self.host, self.user, self.password, self.database)
        c = conn.cursor()
        return c

    def retrieve_users(self, table, statuses_threshold):
        print ('db:retrieve_users' )
        users = []
        #columns = []
        try:
            self.cursor = self.connectMysqlDB()
        except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
        if(self.cursor is not None):
            #sql = "show columns from {0}".format(self.trust_table)
            #self.cursor.execute(sql)
            #columns_name = self.cursor.fetchall()
            #for row in columns_name:
            #        columns.append(row[0])
            sql = "select distinct id from {0} where statuses_count > {1}".format(table, statuses_threshold)
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            for row in result:
                users.append(row[0])

        print ('len(data):  '  , len(users) , '  ' ,  users[0] )
        #trust_data_frame = pd.DataFrame(data = data, columns = columns)
        #trust_data_frame = trust_data_frame.set_index('user_id')
        return users



    def write_to_file(self, data, file_name, thresh = 0):
        f = open(file_name,'w')
        if isinstance(data , list):
            for elem in data:
                f.write(str( str(elem) + '\n'))
        elif isinstance(data, dict):
            for key,val in data.items():
                if abs(val) > thresh:
                    f.write(str( str(key) + '\t,\t' + str(val) + '\n'))
        f.close()

    def cosine_sim(self, delta_tt, delta_tf, delta_tr, delete_list):
        print ('cosine_sim:' )
        print ( delta_tt.shape, ' , ', delta_tf.shape, ' , ', delta_tr.shape )
        cos_sim = []
        cos_sim_r = []
        order = []
        for i in range(0,delta_tt.shape[1]):
            if i in delete_list:
                continue
            data = np.concatenate((delta_tt[:,i].reshape(delta_tt.shape[0],1), delta_tf[:,i].reshape(delta_tf.shape[0],1)), axis = 1)
            data_r = np.concatenate((delta_tt[:,i].reshape(delta_tt.shape[0],1), delta_tr[:,i].reshape(delta_tr.shape[0],1)), axis = 1)
            if data.shape[0] < 1:
                continue
            if i% 10000 == 0:
                print ( 'i: ', i, ' , ',  data.shape, ' , ', data_r.shape )
            cos_sim.append(sk.cosine_similarity(np.transpose(data))[0][1] )
            cos_sim_r.append(sk.cosine_similarity(np.transpose(data_r))[0][1] )
            order.append(i)
        print ('len(cos_sim): ' , len(cos_sim) )
        return cos_sim , cos_sim_r, order


    def friends_lang_avg(self, network , two_hops_users_list, random_friends):
        print ( "calculate_friends_avg" )
        print ( 'len(random_friends): ', len(random_friends) )
        friends_topics_avg = np.zeros( ( len(two_hops_users_list), self.num_of_topics) )
        random_topics_avg = np.zeros( ( len(two_hops_users_list), self.num_of_topics) )
        print ( "len(friends_topics_avg): " , len(friends_topics_avg), ' , ', len(two_hops_users_list) )
        counter = 0
        for user_id in two_hops_users_list:
            if user_id not in random_friends:
                continue
            counter +=1
            user_index = self.all_users_indices[user_id]
            #print user_index
            #continue

            friends = network[user_id]
            friends_indices = [ self.all_users_indices[fid] for fid in friends]
            random_indices = random_friends[user_id]

            friends_topics_avg[user_index ] = np.mean(self.time_user_topic[2, friends_indices, :], axis = 0)
            random_topics_avg [user_index] = np.mean(self.time_user_topic[2 , random_indices, : ] , axis = 0)

        return friends_topics_avg, random_topics_avg



    def random_draw(self, network, three_hops_users_list, two_hops_users_list):
        print ( 'random_draw' )
        #df = trust_df[trust_df.num_of_friends > 1]
        #df = df[np.isfinite(df['diff_score'])]
        #users_id = df.index.tolist()
        #num_3hops_users = trust_df.shape[0]
        #trust_df['avg_friends'] = 0
        #trust_df['avg_random_people'] = 0
        #test_k = 0
        random_friends = {}
        for user_id in two_hops_users_list:
            #if test_k < 3:
            #       print trust_df.loc[user_id]['avg_friends']
            friends = network[user_id]
            #friends_value = trust_df.wo_retweets.loc[friends]
            #friends_avg  = np.mean(friends_value)

            #if friends_avg != friends_avg:
            #        print 'friends_avg: ' , friends_avg, ' , ', len(friends)
            #        continue
            #print 'f-a: ' , friends_avg
            #trust_df.set_value(user_id, 'avg_friends' , friends_avg )
            random_indices = random.sample(range(len(three_hops_users_list)), k = len(friends)) #k=int(df.num_of_friends.loc[user_id]) )
            random_people = [ three_hops_users_list[i] for i in random_indices ]
            random_friends[user_id] = [ self.all_users_indices[rp] for rp in random_people]
            #rp_values = trust_df.wo_retweets.loc[random_people]
            #rp_avg  = np.mean(rp_values)
            #trust_df.set_value(user_id,'avg_random_people', rp_avg)

        #trust_df['diff_friends'] = trust_df.avg_friends - trust_df.first_part
        #trust_df['diff_random_people'] = trust_df.avg_random_people - trust_df.first_part
        return  random_friends



    def retrieve_network(self, two_hops_users_list, three_hops_users_list):
        print ("retrieve_network" )
        network = {}

        updated_all_users_list = set(two_hops_users_list)
        print ('len(updated_all_users_list): ', len(updated_all_users_list) , ' len(two_hops_users_list): ', len(two_hops_users_list))
        try:
            self.cursor = self.connectMysqlDB()
        except:
            print("error while connecting to database:", sys.exc_info()[0])
            raise
        if(self.cursor is not None):
            ####  save network as adjacency lists. A dictionary of nodes, in which For each node we keep its neighbors in a list
            network = { two_hops_users_list[i] : [] for i in xrange(len(two_hops_users_list))}
            #### for each node in two hops distance, we fetch its neighbors from db.
            for i in range(len(two_hops_users_list)):
                user_id = two_hops_users_list[i]
                sql = "select * from {0} where source_id={1}".format(self.network_table, user_id)
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                user_index = i
                ls = []
                for row in result:
                    #### only add the neighbors, for which their language info is available
                    if  row[2] in self.all_users_indices:
                        user2 = self.all_users_indices[row[2]]
                        #network[user_id].append(row[2])
                        ls.append(row[2])
                ls  = random.sample(ls, min(len(ls), 100) )
                network[user_id] = ls
                updated_all_users_list = updated_all_users_list.union(set(ls))
        print ('len(updated_all_users_list): ', len(updated_all_users_list) )
        return network, list(updated_all_users_list)

    def calculate_deltas(self, network, two_hops_users_list, friends_topics_avg, random_topics_avg):
        print ("calculate_deltas" )
        delete_mask = []
        delta_tt = np.zeros(( len(two_hops_users_list), self.num_of_topics))
        delta_tf = np.zeros(( len(two_hops_users_list), self.num_of_topics))
        delta_tr = np.zeros(( len(two_hops_users_list), self.num_of_topics))
        for user_id in two_hops_users_list:
            user_index = self.all_users_indices[user_id]
            delta_tt[user_index] = np.subtract(self.time_user_topic[1,user_index,:] , self.time_user_topic[0,user_index,:] )
            delta_tf[user_index] = np.subtract(friends_topics_avg[user_index] , self.time_user_topic[0,user_index,:] )
            delta_tr[user_index] = np.subtract(random_topics_avg[user_index] , self.time_user_topic[0,user_index, :] )
            if len(network[user_id]) < 6:
                    delete_mask.append(user_index)
        print ('shapes: ' )
        print (delta_tt.shape )
        print (delta_tf.shape )
        print (delta_tr.shape )

        delta_tt = np.delete(delta_tt, delete_mask, 0)
        delta_tf = np.delete(delta_tf, delete_mask, 0)
        delta_tr = np.delete(delta_tr, delete_mask, 0)
        print ("mask_len: " , len(delete_mask) )

        print ("self.delta_tt.shape: ", delta_tt.shape )
        print ("self.delta_tf.shape: ", delta_tf.shape )
        print ("self.delta_tr.shape: ", delta_tr.shape )
        return delta_tt, delta_tf, delta_tr

    def retrieve_topics(self, two_hops_users_list, updated_all_users_list):
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
                sql = "select * from {0} where group_id in ({1})".format(self.feat_tables[i], ','.join([str(k) for k in updated_all_users_list]) )
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
                counter = 0
                ll = len(result)
                print ('len(result): ', ll)
                for row in result:
                    counter+=1
                    if counter % 1000000 == 0:
                        print ( 'counter: ' , counter, ' , ', ll )
                    #### drop rows that users are not in all_users_indices list
                    if ( row[1] not  in self.all_users_indices ):
                        continue
                    #### for feature tables 0 and 1, drop those rows that users are not in two hops distance
                    if ( i < 2) and  ( row[1] not in two_hops_users_list):
                        continue
                    #print ('row: ', row)
                    #### fill time_user_topic array. where by 'topic' here we mean 'word'
                    user = self.all_users_indices[row[1]]
                    #topic = int(row[2])
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



ts = Trust_script()
ts.main()