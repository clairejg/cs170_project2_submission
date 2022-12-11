import numpy as np
#import numpy to deal with creating and modify Multidimentsional array. Source=>
#https://linuxhint.com/python_numpy_tutorial/
#https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html

import os
#os :https://medium.com/@dagyeom23658/meaning-of-os-getcwd-os-path-join-ccfa0403137a
#Python requires a lot of code related to file paths or directories. 
# All functions related to file and directory paths use the os module, 
# so import of the os module is required.

import math
import random
import time

def main():
    print( "\n<Welcome to Claire Feature Selection Algorithm>\n")
    print("Type in the name of the file to test: ")
    text_file_name = str(input())
    text_file_path= os.path.join(os.getcwd(), text_file_name)
    text_opened = np.genfromtxt(text_file_path)
    

    print("\nType the number of the algorithm you want to run.")
    print("1. Forward Selection")
    print("2. Backward Elimination" )
    algorithm = int(input())
    instances=0
    instances=text_opened.shape[0]
    print("This dataset has '{}' features (not including the class attribute), with '{}' instances\n".format(text_opened.shape[1]-1, instances))
    print("Running nearest neighbor with all {} features, using \"leaving-one-out\" evalutation, I get an accuracy of {}%\n".format(text_opened.shape[1]-1, leave_one_evaluation(text_opened,instances)))
    print('Beginning search...\n')

    if algorithm == 1:
        return forward_selection(text_opened)
    elif algorithm == 2:
        return backward_elimination(text_opened)


def leave_one_evaluation(text_opened,instance):
    first_feature_list_count = 0
    second_feature_list_count = 0
    for i in range(instance):
        if text_opened[i][0] == 1:
            first_feature_list_count+= 1
        else:
            second_feature_list_count += 1

    if first_feature_list_count > second_feature_list_count:
         accuracy_percentage= (first_feature_list_count / instance) * 100.0
    else:
         accuracy_percentage = (second_feature_list_count/ instance) * 100.0

    return accuracy_percentage



def forward_selection(text_file_opened):
    start = time.time()
    num_of_features = text_file_opened.shape[1] #returns number of columns of the file== number of features.
    curr_subset_features = [] 
    best_subset_features = [] 
    best_accuracy = 0
    
    for i in range(1, num_of_features): 
        feature_to_add = 0  # add to this set only if it is not addes yet. Add it only once.
        temp_best_accuracy = 0 

        for k in range(1, num_of_features): # runs through the features and calculates accuracy based on the current set with the new addition
            if k not in curr_subset_features:
                
                accuracy_found = leave_one_out_cross_validation(curr_subset_features, text_file_opened, k, 1)                
                printout_features=list(curr_subset_features)+[k]
                #"{:.1f}".format(accuracy_found)
                print("*Using feature(s) {} accuracy is {}%".format(printout_features, round((accuracy_found*100),1)))

                if accuracy_found> temp_best_accuracy:
                    temp_best_accuracy = accuracy_found
                    feature_to_add = k
        curr_subset_features.append(feature_to_add) 

        #IF ELSE statement: to see if there is Accuaracy decrease.
        if temp_best_accuracy >= best_accuracy: 
            best_accuracy = temp_best_accuracy
            best_subset_features = list(curr_subset_features)
            print("=>Feature set {} was best, accuracy is {}%\n".format(curr_subset_features,  round((temp_best_accuracy*100),1)))
        else:
            print("=>Feature set {} was best, accuracy is {}%\n".format(curr_subset_features, round((temp_best_accuracy*100),1)))
            #"{:.1f}".format(best_accuracy)
    print("Finished search!! The best feature subset is {}, which has an accuracy of {}%".format( best_subset_features, round((best_accuracy*100),1)))

    end = time.time()
    consumed_time=(end-start)
    print("Time to finish: %s" %round(consumed_time,1))
    return

def backward_elimination(text_file_open):
    start = time.time()
    num_of_features = text_file_open.shape[1]
    best_subset_features=[]
    best_accuracy = 0
    
    curr_subset_features=list(range(1,num_of_features))

    for i in range(1, num_of_features ):
        feature_delete = []
        temp_best_accuracy = 0.
        for k in range(1, num_of_features ):
            if k not in  curr_subset_features:
                    continue
            check_features = [n for n in curr_subset_features if n != k]
            found_accuracy = leave_one_out_cross_validation( curr_subset_features, text_file_open, k, 2)
            features=check_features
            if(features==[]):
                print ("Done searching.")
                #print("Finished search!! The best feature subset is {}, which has an accuracy of {}%".format( best_subset_features, round((best_accuracy*100),1)))
    
            else:
                print("*Using feature(s) {} accuracy is {}%".format(features, round((found_accuracy*100),1)))

            if found_accuracy > temp_best_accuracy:
               temp_best_accuracy = found_accuracy
               feature_delete = k

        #if feature_delete:
        if feature_delete in curr_subset_features:
            curr_subset_features.remove(feature_delete)
            if temp_best_accuracy >= best_accuracy:
                best_accuracy = temp_best_accuracy
                best_subset_features = list(curr_subset_features)
                
                if(features==[]):
                    print ("")
                else:
                    print("=>Feature set {} was best, accuracy is {}%\n".format(curr_subset_features,  round((temp_best_accuracy*100),1)))
            else:
                if(features==[]):
                    print ("")
                else:
                 
                    print("=>Feature set {} was best, accuracy is {}%\n".format(curr_subset_features, round((temp_best_accuracy*100),1)))
            #"{:.1f}".format(best_accuracy)
    print("Finished search!! The best feature subset is {}, which has an accuracy of {}%".format( best_subset_features, round((best_accuracy*100),1)))
    end = time.time()
    consumed_time=(end-start)
    print("Time to finish: %s" %round(consumed_time,1))
    return


#finds the accuracy using Euclidean distance.
def leave_one_out_cross_validation(curr_subset_feat, curr_text_file, curr_checking_feature, algorithm):
    #euclidean_d = 0
    curr_subset_checking_feature = list(curr_subset_feat)
    if algorithm == 1: 
        curr_subset_checking_feature.append(curr_checking_feature)

    if algorithm == 2: 
       curr_subset_checking_feature.remove(curr_checking_feature)
    number_correctly_classfied = 0
    curr_min_d = math.inf #infinity
    result = 0 

    for a in curr_text_file:
        curr_min_d = math.inf
        for b in curr_text_file:
            if not np.array_equal(b, a): 
                euclidean_d = 0 #must set to 0 at here to reset
                for j in curr_subset_checking_feature:
                    euclidean_d += pow((a[j] - b[j]), 2) 
                    # to get euclidian distance
                    # https://www.programiz.com/python-programming/methods/built-in/pow

                euclidean_distance=  math.sqrt(euclidean_d)   
                if euclidean_distance < curr_min_d:
                    curr_min_d = euclidean_distance
                    result = b[0] 
        if result == a[0]:
            number_correctly_classfied += 1
        
        returing_accuracy = number_correctly_classfied / (len(curr_text_file))
    return returing_accuracy



if __name__ == '__main__':
    main()