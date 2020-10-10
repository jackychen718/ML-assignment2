from util import split_train_test_spam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlrose_hiive as mlrose
'''
train_features_spam,train_labels_spam,test_features_spam,test_labels_spam=split_train_test_spam()
scaler=StandardScaler()
scaler.fit(train_features_spam)
train_features_spam_norm=scaler.transform(train_features_spam)
test_features_spam_norm=scaler.transform(test_features_spam)
'''
def relu(z):
    z[z<0]=0
    return z

def sigmoid(z):
    reduced_z=z[:,0]
    result=1./(1.+np.exp(-reduced_z))
    return result

count=0

def spam_nn_fit(state,train_features,train_labels):
    global count
    count+=1
    state= (state-500)*1.e-2
    w1=state[0:228].reshape((57,4))
    b1=state[228:232].reshape((1,4))
    w2=state[232:236].reshape((4,1))
    b2=state[236:237].reshape((1,1))
    m=len(train_features)
    A1=relu(np.dot(train_features,w1)+b1)
    proba=sigmoid(np.dot(A1,w2)+b2)
    pos=np.sum(train_labels*np.log(proba+1.e-6))
    neg=np.sum((1-train_labels)*np.log(1-proba+1.e-6))
    loss=(pos+neg)/m
    return loss

def predict(state_best,test_features_norm):
    state= (state_best-500)*1.e-2
    w1=state[0:228].reshape((57,4))
    b1=state[228:232].reshape((1,4))
    w2=state[232:236].reshape((4,1))
    b2=state[236:237].reshape((1,1))
    A1=relu(np.dot(test_features_norm,w1)+b1)
    Z2=np.dot(A1,w2)+b2
    result=Z2[:,0]>0
    return result.astype(int)
'''
state=np.random.rand(237)*1000
print(state)
loss=spam_nn_fit(state)
print(loss)
'''


def rhc(opt):
    global count
    count=0
    rhc_loss_list=[]
    rhc_train_acc_list=[]
    rhc_test_acc_list=[]
    for num_restart in range(0,101,5):
        best_state_spam,best_fitness_spam,_=mlrose.random_hill_climb(opt,restarts=num_restart,curve=False)
        train_predict_hill=predict(best_state_spam,train_features_spam_norm)
        test_predict_hill=predict(best_state_spam,test_features_spam_norm)
        train_accuracy_hill=accuracy_score(train_labels_spam,train_predict_hill)
        test_accuracy_hill=accuracy_score(test_labels_spam,test_predict_hill)
        rhc_loss_list.append(best_fitness_spam)
        rhc_train_acc_list.append(train_accuracy_hill)
        rhc_test_acc_list.append(test_accuracy_hill)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(np.arange(0,101,5),rhc_loss_list,label='-1*loss')
    plt.xlabel('random restart num')
    plt.ylabel('minus of loss')
    plt.title('loss versus restart num')
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(np.arange(0,101,5),rhc_train_acc_list,label='train')
    plt.plot(np.arange(0,101,5),rhc_test_acc_list,label='test')
    plt.xlabel('random restart num')
    plt.ylabel('accuracy')
    plt.title('accuracy versus restart num')
    plt.legend(loc='lower right')
    plt.show()
    
def gen_alg(opt):
    global count
    count=0
    gen_loss_list=[]
    gen_train_acc_list=[]
    gen_test_acc_list=[]
    for population in range(200,4001,200):
        best_state_spam,best_fitness_spam,_=mlrose.genetic_alg(opt,pop_size=population,curve=True)
        train_predict_gen=predict(best_state_spam,train_features_spam_norm)
        test_predict_gen=predict(best_state_spam,test_features_spam_norm)
        train_accuracy_hill=accuracy_score(train_labels_spam,train_predict_gen)
        test_accuracy_hill=accuracy_score(test_labels_spam,test_predict_gen)
        gen_loss_list.append(best_fitness_spam)
        gen_train_acc_list.append(train_accuracy_hill)
        gen_test_acc_list.append(test_accuracy_hill)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(np.arange(200,4001,200),gen_loss_list,label='-1*loss')
    plt.xlabel('population num')
    plt.ylabel('minus of loss')
    plt.title('loss versus population num')
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(np.arange(200,4001,200),gen_train_acc_list,label='train')
    plt.plot(np.arange(200,4001,200),gen_test_acc_list,label='test')
    plt.xlabel('population num')
    plt.ylabel('accuracy')
    plt.title('accuracy versus population num')
    plt.legend(loc='lower right')
    plt.show()
        
def s_ann(opt):
    global count
    count=0
    ann_loss_list=[]
    ann_train_acc_list=[]
    ann_test_acc_list=[]
    for exp_value in np.arange(0.0005,0.0101,0.0005):
        best_state_spam,best_fitness_spam,_=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(exp_const=exp_value),curve=True)
        train_predict_gen=predict(best_state_spam,train_features_spam_norm)
        test_predict_gen=predict(best_state_spam,test_features_spam_norm)
        train_accuracy_hill=accuracy_score(train_labels_spam,train_predict_gen)
        test_accuracy_hill=accuracy_score(test_labels_spam,test_predict_gen)
        ann_loss_list.append(best_fitness_spam)
        ann_train_acc_list.append(train_accuracy_hill)
        ann_test_acc_list.append(test_accuracy_hill)
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(np.arange(0.0005,0.0101,0.0005),ann_loss_list,label='-1*loss')
    plt.xlabel('exp_const value')
    plt.ylabel('minus of loss')
    plt.title('loss versus exp_const value')
    plt.legend(loc='lower right')
    plt.subplot(122)
    plt.plot(np.arange(0.0005,0.0101,0.0005),ann_train_acc_list,label='train')
    plt.plot(np.arange(0.0005,0.0101,0.0005),ann_test_acc_list,label='test')
    plt.xlabel('exp_const value')
    plt.ylabel('accuracy')
    plt.title('accuracy versus exp_const value')
    plt.legend(loc='lower right')
    plt.show()


def method_compare(opt,train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam):
    #best_state_spam,best_fitness_spam,fitness_curve=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(),curve=True)
    #best_state_spam,best_fitness_spam,fitness_curve=mlrose.genetic_alg(opt,pop_size=2000,curve=True)

    global count
    count=0
    loss_list_rhc=[]
    loss_list_ann=[]
    loss_list_ga=[]
    
    train_acc_list_rhc=[]
    train_acc_list_ann=[]
    train_acc_list_ga=[]
    test_acc_list_rhc=[]
    test_acc_list_ann=[]
    test_acc_list_ga=[]
    
    fitness_call_list_rhc=[]
    fitness_call_list_ann=[]
    fitness_call_list_ga=[]
    # ten rounds of rhc
    for i in range(10):
        count=0
        best_state_spam,best_fitness_spam,fitness_curve=mlrose.random_hill_climb(opt,restarts=70,curve=True)
        loss_list_rhc.append(best_fitness_spam)
        train_predict_rhc=predict(best_state_spam,train_features_spam_norm)
        test_predict_rhc=predict(best_state_spam,test_features_spam_norm)
        train_acc_list_rhc.append(accuracy_score(train_labels_spam,train_predict_rhc))
        test_acc_list_rhc.append(accuracy_score(test_labels_spam,test_predict_rhc))
        fitness_call_list_rhc.append(count)
    #ten rounds of simulated annealing
    for i in range(10):
        count=0
        best_state_spam,best_fitness_spam,_=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(exp_const=0.003),curve=True)
        loss_list_ann.append(best_fitness_spam)
        train_predict_ann=predict(best_state_spam,train_features_spam_norm)
        test_predict_ann=predict(best_state_spam,test_features_spam_norm)
        train_acc_list_ann.append(accuracy_score(train_labels_spam,train_predict_ann))
        test_acc_list_ann.append(accuracy_score(test_labels_spam,test_predict_ann))
        fitness_call_list_ann.append(count)
    #ten rounds of genetic algorithm
    for i in range(10):
        count=0
        best_state_spam,best_fitness_spam,_=mlrose.genetic_alg(opt,pop_size=1000,curve=True)
        loss_list_ga.append(best_fitness_spam)
        train_predict_ga=predict(best_state_spam,train_features_spam_norm)
        test_predict_ga=predict(best_state_spam,test_features_spam_norm)
        train_acc_list_ga.append(accuracy_score(train_labels_spam,train_predict_ga))
        test_acc_list_ga.append(accuracy_score(test_labels_spam,test_predict_ga))
        fitness_call_list_ga.append(count)
    
    #plot loss curve
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(1,11),loss_list_rhc,label='rhc')
    plt.plot(np.arange(1,11),loss_list_ann,label='s_ann')
    plt.plot(np.arange(1,11),loss_list_ga,label='ga')
    plt.xlabel('rounds')
    plt.ylabel('-1*losss')
    plt.title('loss versus different algorithm')
    plt.legend()
    plt.show()
    
    #plot acc curve 
    plt.figure(figsize=(15,6))
    plt.subplot(131)
    plt.plot(np.arange(1,11),train_acc_list_rhc,label='train')
    plt.plot(np.arange(1,11),test_acc_list_rhc,label='test')
    plt.xlabel('rounds')
    plt.ylabel('accuracy')
    plt.title('rhc')
    plt.legend()
    plt.subplot(132)
    plt.plot(np.arange(1,11),train_acc_list_ann,label='train')
    plt.plot(np.arange(1,11),test_acc_list_ann,label='test')
    plt.xlabel('rounds')
    plt.ylabel('accuracy')
    plt.title('simulated annealing')
    plt.legend()
    plt.subplot(133)
    plt.plot(np.arange(1,11),train_acc_list_ga,label='train')
    plt.plot(np.arange(1,11),test_acc_list_ga,label='test')
    plt.xlabel('rounds')
    plt.ylabel('accuracy')
    plt.title('genetic algorithm')
    plt.legend()
    
    #plot fitness call
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(1,11),fitness_call_list_rhc,label='rhc')
    plt.plot(np.arange(1,11),fitness_call_list_ann,label='s_ann')
    plt.plot(np.arange(1,11),fitness_call_list_ga,label='ga')
    plt.xlabel('rounds')
    plt.ylabel('fitness call number')
    plt.title('fitness call num versus different algorithm')
    plt.legend()
    plt.show()
    
    
def scale_ann(train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam):
    global count
    train_acc_list=[]
    test_acc_list=[]
    fitness_call_list=[]
    loss_list=[]
    for i in range(400,4001,400):
        count=0
        train_features_sub=train_features_spam_norm[:i,:]
        train_labels_sub=train_labels_spam[:i]
        fitness_obj=mlrose.CustomFitness(spam_nn_fit,train_features=train_features_sub,train_labels=train_labels_sub)
        opt=mlrose.DiscreteOpt(237,fitness_obj,maximize=True,max_val=1001)
        best_state_spam,best_fitness_spam,_=mlrose.simulated_annealing(opt,schedule=mlrose.ExpDecay(exp_const=0.003),curve=True)
        loss_list.append(best_fitness_spam)
        train_predict=predict(best_state_spam,train_features_sub)
        test_predict=predict(best_state_spam,test_features_spam_norm)
        fitness_call_list.append(count)
        train_acc_list.append(accuracy_score(train_labels_sub,train_predict))
        test_acc_list.append(accuracy_score(test_labels_spam,test_predict))
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.plot(np.arange(400,4001,400),loss_list,label='-1*loss')
    plt.xlabel('training size')
    plt.ylabel('-1*loss')
    plt.title('loss versus training size')
    plt.legend()
    plt.subplot(122)
    plt.plot(np.arange(400,4001,400),train_acc_list,label='train')
    plt.plot(np.arange(400,4001,400),test_acc_list,label='test')
    plt.xlabel('training size')
    plt.ylabel('accuracy')
    plt.title('accuracy versus training size')
    plt.legend()
    plt.show()
    # fitness calls versus training size
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(400,4001,400),fitness_call_list,label='#.calls')
    plt.xlabel('training size')
    plt.ylabel('fitness calls')
    plt.legend()
    plt.show()
    

        
        
    
if __name__=="__main__":
    train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam=split_train_test_spam()
    fitness_obj=mlrose.CustomFitness(spam_nn_fit,train_features=train_features_spam_norm,train_labels=train_labels_spam)
    opt=mlrose.DiscreteOpt(237,fitness_obj,maximize=True,max_val=1001)
    rhc(opt)
    s_ann(opt)
    gen_alg(opt)
    method_compare(opt,train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam)
    scale_ann(train_features_spam_norm,train_labels_spam,test_features_spam_norm,test_labels_spam)
    
    
    