# Artificial Intelligence II - file 01

# Amin Alhawary

from math import cos, pi
from random import randint, uniform
import matplotlib.pyplot as plt
import numpy as np

p = 50
n = 20
g = 100

# Rosenbrock limits - Uncomment below if using
upperLimit = 100
lowerLimit = -100

# Zakharov limits - Uncomment below if using
# upperLimit = 10
# lowerLimit = -5

global mutstep
global mutrate

mutStepDiff = (upperLimit - lowerLimit) * 0.05
mutRateDiff = 1

def Rosenbrock(indiv): # Rosenbrock function
    sum = 0
    n = len(indiv.gene)
    for i in range(0,n-1):
        x = indiv.gene[i]
        x2 = indiv.gene[i+1]
        sum = sum + 100 * ((x2 - x ** 2)**2) + ((1 - x)**2)

    return sum

def Zakharov(indiv): # Zakharov function
    sum = 0
    d = len(indiv.gene)
    for x in indiv.gene:
        sum = sum + x**2
        
    temp = 0
    for i in range(0,d):
        x = indiv.gene[i]
        temp = temp + 0.5 * i * x

    sum = sum + temp ** 2 + temp ** 4

    return sum 

def myformat(x): # Formatting to make numbers more readable
    if x>500000000:
        return "{}B".format(round(x/1000000000,2))
    elif x>500000:
        return "{}M".format(round(x/1000000,2))
    elif x> 5000:
        return "{}K".format(round(x/1000,2))
    else: return round(x,2)

class individual: # Individual Class
    def __init__(self):
        self.gene = np.zeros(n,dtype=float)
        self.fitness = 0

    def __str__(self):
        return """
        Individual Class
        Fitness = {}""".format(round(self.fitness,2))

    def inherit(self, gene): # Method to make inheritance of genes easier
        self.gene = gene
        self.getFitness()
        
    def getFitness(self): # Method where fitness is calculated
        self.fitness = Rosenbrock(self)
        # self.fitness = Zakharov(self)

    def _randomise(self): # Used when initialising the population
        self.gene = np.random.uniform(lowerLimit,upperLimit, n)
        self.getFitness()

class population(): # Population Class
    def __init__(self):
        self.pop = []
        self.fitness = 0
        self.gen = 0
        self.worst = individual() # Worst fitness set to zero to facilitate comparison
        self.best = individual()
        self.best.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double)) # Best fitness is set to highest value to facilitate comparison
        self.average = 0
        self.mutCount = 0

        # Generation 0, initialisation

        for i in range(0,p):
            temp = individual()
            temp._randomise()

            # Add randomised individual to population
            self.pop.append(temp)
            self.fitness = self.fitness + temp.fitness

            # Finding best and worst 
            if temp.fitness > self.worst.fitness:
                self.worst = temp
            if temp.fitness < self.best.fitness:
                self.best = temp
        self.average = self.fitness/p

    def fill(self,population): # Similar to inherit() method in individual class but for population
        self.pop = []        
        self.fitness = 0
        self.gen = 0
        self.worst = individual()
        self.best = individual()
        self.best.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double))
        self.average = 0
        self.mutCount = 0
        
        count = 0
        for indiv in population:
            temp = individual()
            temp.inherit(indiv.gene)
            self.pop.append(temp)
            self.fitness = self.fitness + temp.fitness
            count = count+1
            
            if temp.fitness > self.worst.fitness:
                self.worst = temp
            if temp.fitness < self.best.fitness:
                self.best = temp
        
        for i in range(count,p):
            temp = individual()
            temp._randomise()
            self.pop.append(temp)
            self.fitness = self.fitness + temp.fitness
            if temp.fitness > self.worst.fitness:
                self.worst = temp
            if temp.fitness < self.best.fitness:
                self.best = temp

        self.average = self.fitness/p


    def _selectIndiv(self): # Selection
        # Tournament Selection
        r1= randint(0, p-1)
        r2 = randint(0, p-1)
        if self.pop[r1].fitness < self.pop[r2].fitness:
            return self.pop[r1]
        else:
            return self.pop[r2]

    def crossover(self): # Crossover

        x = self._selectIndiv() # Tournament selection as mentioned before
        y = self._selectIndiv() # Tournament selection as mentioned before
        
        a = individual()
        b = individual() 
        
        # Single point crossover  - Uncomment below if using

        # upper = int(0.75*n)
        # lower = int(0.25*n)

        # crossPoint = randint(lower,upper)

        # gene1 = []
        # gene2 = []

        # for i in range(0,crossPoint):
        #         gene1.append(x.gene[i])
        #         gene2.append(y.gene[i])
        # for i in range(crossPoint,n):
        #         gene1.append(y.gene[i])
        #         gene2.append(x.gene[i])

        #########

        # Arithmetic Crossover  - Uncomment below if using
        alpha = 0.65

        gene1 = []
        gene2 = []

        for i in range(0,n):
            temp1 = x.gene[i] + ((1 - alpha) * y.gene[i])
            temp2 = y.gene[i] - (alpha * x.gene[i])
            gene1.append(temp1)
            gene2.append(temp2)

        # Done with crossover here 

        a.inherit(gene1)
        b.inherit(gene2)
        self._mutation(a)
        self._mutation(b)

        if a.fitness > self.newWorst.fitness:
            self.newWorst = a
        if self.newBest.fitness > a.fitness:
            self.newBest = a
        
        if b.fitness > self.newWorst.fitness:
            self.newWorst = b
        if self.newBest.fitness > b.fitness:
            self.newBest = b

        self.fitness = self.fitness + a.fitness + b.fitness
        
        self.newPop.append(a)
        self.newPop.append(b)

    def newGen(self):
        self.fitness = 0
        self.average = 0
        self.newBest = individual()
        self.newBest.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double))
        self.newWorst = individual()
        self.newPop = []
        
        for i in range(0,p,2):
            self.crossover()
        self.average = self.fitness/p

        for indiv in self.newPop:
            if indiv == self.worst:
                indiv = self.best

        self.worst = self.newWorst

        if self.best.fitness > self.newBest.fitness:
                self.best = self.newBest
        
        self.pop = self.newPop
        self.gen = self.gen + 1

    def _mutation(self, indiv, i=-1):
        global mutrate
        global mutstep

        if i == -1:
            for i in range(n):
                mutprob = randint(0,100)
                if mutprob<mutrate:
                    mutation= uniform(-mutstep,mutstep)
                    newChromosome = indiv.gene[i] + mutation
                    if newChromosome < lowerLimit or newChromosome > upperLimit:
                        self._mutation(indiv,i)
                    else: 
                        self.mutCount = self.mutCount+1
                        indiv.gene[i] = newChromosome
        else:
            mutprob = randint(0,100)
            if mutprob<mutrate:
                mutation= uniform(-mutstep,mutstep)
                newChromosome = indiv.gene[i] + mutation
                if newChromosome < lowerLimit or newChromosome > upperLimit:
                    self._mutation(indiv,i)
            
class experiment():
    def __init__(self,ppl):

        self.ppl = ppl
        self.bestArray = []
        self.worstArray = []
        self.avrgArray = []

        self.page = 0
        self.currentbest = []

        dump = individual()
        dump.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double))
        self.currentbest.append(dump.fitness)

        self.bestArray.append(self.ppl.best.fitness)
        self.worstArray.append(self.ppl.worst.fitness)
        self.avrgArray.append(self.ppl.average)

    def findHML(self,x,y):
        diff = y-x
        lowerQuart = round(x + (diff * 0.25),2)
        upperQuart = round(x + (diff * 0.75),2)
        return [x,lowerQuart,upperQuart,y]

    def start(self):
            print("Starting bounds:\nSteps:",0.0125*(upperLimit-lowerLimit),upperLimit/2,"\nRates",1,50)
            return self.perform(self.ppl,0.0125*(upperLimit-lowerLimit),upperLimit/2,1,50)

    def perform(self,ppl,lowerStep,upperStep,lowerRate,upperRate):
            
            # A sweep of High, Medium, Low values for Mutation Step and Mutation Rate inbetween two limits for each.
            # Each combination of High/Medium/Low Rates and Steps are compared and best results are "zoomed" into

            global mutrate
            global mutstep

            stepBoundries = self.findHML(lowerStep,upperStep)
            rateBoundries = self.findHML(lowerRate,upperRate)

            steps = []
            rates = []

            for i in range(3,0,-1): 
                steps.append(uniform(stepBoundries[i],stepBoundries[i-1]))

            for i in range(3,0,-1): 
                rates.append(uniform(rateBoundries[i],rateBoundries[i-1]))
            
            fig, ax = plt.subplots(3,3,figsize=(12,12)) 
            plt.set_cmap('jet')

            # Title - Uncomment applicable
            fig.suptitle("Rosenbrock Function\nRates {} - {}, Steps {} - {}".format(round(lowerRate,2),round(upperRate,2),round(lowerStep,2),round(upperStep,2)))
            # fig.suptitle("Zakharov Function\nRates {} - {}, Steps {} - {}".format(round(lowerRate,2),round(upperRate,2),round(lowerStep,2),round(upperStep,2)))

            self.bests = []
            for i in range(3):
                self.bests.append([])
            
            i = -1
            j = -1
            print("_____________")
            print("Rates - H: {}, M: {}, L: {}".format(round(rates[0],2),round(rates[1],2),round(rates[2],2)))
            print("Steps - H: {}, M: {}, L: {}".format(round(steps[0],2),round(steps[1],2),round(steps[2],2))) 
            print("_____________")
            for x in rates:
                i=i+1
                j=-1
                for y in steps:
                    j=j+1
                    mutstep = y
                    mutrate = x
                    
                    bestaverage = 0
                    avrgavrg = 0
           
                    word = ["High","Medium","Low"]
                    print("Mutrate = {}% ({})".format(mutrate,word[i]))
                    print("Mutstep = {} ({})".format(mutstep,word[j]))

                    k = 1
                    for k in range(1,5):
                        pplCopy = population()
                        pplCopy.fill(ppl.pop)
                        self.bestArray = []
                        self.worstArray = []
                        self.avrgArray = []

                        self.bestArray.append(pplCopy.best.fitness)
                        self.worstArray.append(pplCopy.worst.fitness)
                        self.avrgArray.append(pplCopy.average)
                        for l in range(1,g):
                            pplCopy.newGen()
                            self.bestArray.append(pplCopy.best.fitness)
                            self.worstArray.append(pplCopy.worst.fitness)
                            self.avrgArray.append(pplCopy.average)

                        bestaverage = bestaverage + pplCopy.best.fitness
                        avrgavrg = avrgavrg + pplCopy.fitness

                    bestaverage = bestaverage/(k+1)
                    avrgavrg = avrgavrg/(k+1)

                    print("Average Best is ", myformat(bestaverage))
                    print("Average Average is ", myformat(avrgavrg))
                    print("---")

                    ax[i,j].plot(self.avrgArray, linewidth=2.0, label='Average')
                    ax[i,j].plot(self.bestArray, linewidth=2.0, label='Best')
                    
                    ax[i,j].set_xlabel("Mutstep = {}, Mutrate = {}%\nMutations = {}\nAverage best of {} runs = {}".format(round(mutstep,2),round(mutrate,2),pplCopy.mutCount,k+1,myformat(bestaverage)))
                    ax[i,j].xaxis.set_label_position('top')
                    ax[i,j].legend()

                    self.bests[i].append(bestaverage)
                

            self.page = self.page + 1

            fig.savefig("img{}.png".format(self.page))

            dump = individual()
            dump.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double))
            temp = dump.fitness

            if ((steps[0] - steps[2]) < mutStepDiff) and ((rates[0] - rates[2]) < mutRateDiff):
                plt.show()
                for i in range(0,3):
                    for j in range(0,3):
                        if self.bests[i][j] < temp:
                            temp = self.bests[i][j]
                            rate = rates[i]
                            step = steps[j]
         
                print("----------------------------------------------------------------------")

                results = []
                results.append([temp, rate, step])
                results.append(self.currentbest)

                return results
            else:
                original = []
                original.append(lowerStep)
                original.append(upperStep)
                original.append(lowerRate)
                original.append(upperRate)

                for i in range(0,3):
                    for j in range(0,3):

                        if self.bests[i][j] < temp:
                            temp = self.bests[i][j]

                            # If this is the best mutrate and mutstep  (yet), ratio between square and next square is calculated
                            # New limit will be between ratio * difference and the direction it should head
                            # Ratio is then used as "bias", the better the rate/step is the more the new limit will try to cover it
                            # If ratio is more than 1 then it is worse than the next square, so pass
                             
                            if i == 0:
                                ratio = self.bests[0][j]/self.bests[1][j] 
                                if ratio > 1:
                                    pass
                                else:
                                    diff = rates[0] - rates[1]
                                    upperRate = original[3]
                                    lowerRate = upperRate - (ratio * diff)
                                    print(ratio,diff)
                                    print("Rate: ", upperRate,lowerRate)

                            elif i == 1:
                                ratio1 = self.bests[0][j]/self.bests[1][j]
                                ratio2 = self.bests[2][j]/self.bests[1][j]
                                if ratio1 < 1 and ratio2 < 1:
                                    diff1 = rates[0] - rates[1]
                                    diff2 = rates[1] - rates[2]
                                    upperRate = original[3] - (ratio1 * diff1)
                                    lowerRate = original[2] + (ratio2 * diff2)
                                    print(ratio1,diff1)
                                    print(ratio2,diff2)
                                    print("Rate: ", upperRate,lowerRate)
                            elif i == 2:
                                ratio = self.bests[2][j]/self.bests[1][j]
                                if ratio > 1:
                                    pass
                                else:                                
                                    diff = rates[1] - rates[2]
                                    lowerRate = original[2]
                                    upperRate = lowerRate + (ratio * diff)
                                    print(ratio,diff)
                                    print("Rate: ", upperRate,lowerRate)

                            if j == 0:
                                ratio = self.bests[i][0]/self.bests[i][1]
                                if ratio > 1:
                                    pass
                                else:
                                    diff = steps[0] - steps[1]
                                    upperStep = original[1]
                                    lowerStep = upperStep - (ratio * diff)
                                    print(ratio,diff)
                                    print("Step: ", upperStep,lowerStep)
                            elif j == 1:
                                ratio1 = self.bests[i][0]/self.bests[i][1]                                
                                ratio2 = self.bests[i][2]/self.bests[i][1]
                                if ratio1 < 1 and ratio2 < 1:
                                    diff1 = steps[0] - steps[1]
                                    diff2 = steps[1] - steps[2]
                                    upperStep = original[1] - (ratio1 * diff1)
                                    lowerStep = original[0] + (ratio2 * diff2)
                                    print(ratio1,diff1)
                                    print(ratio2,diff2)
                                    print("Step: ", upperStep,lowerStep)
                            elif j == 2:
                                ratio = self.bests[i][2]/self.bests[i][1]
                                if ratio>1:
                                    pass
                                else:
                                    diff = steps[1] - steps[2]
                                    lowerStep = original[0]
                                    upperStep = lowerStep + (ratio * diff)
                                    print(ratio,diff)
                                    print("Step: ", upperStep,lowerStep)

                            # Compare with all results
                            if temp < self.currentbest[0]:
                                self.currentbest = [temp,rates[i],steps[j]]
                print("New bounds:\nSteps:",lowerStep,upperStep,"\nRates",lowerRate,upperRate)
                return self.perform(ppl,lowerStep,upperStep,lowerRate,upperRate)

ppl = population()
x = experiment(ppl)
modification = x.start()

print("""

Final Average of bests: {}
found at rate = {}, step = {}

Best Average of bests: {}
found at rate = {}, step = {}

""".format(modification[0][0],modification[0][1],modification[0][2],modification[1][0],modification[1][1],modification[1][2]))

print("""
----------------------------------------------------------------------

Further tests:
""")

for k in range(0,2):
    fig, ax = plt.subplots(3,3,figsize=(12,12)) 
    plt.set_cmap('jet')
    fig.suptitle("Test #{}, Mutrate = {}%, Mutstep = {}".format(k+1,round(mutrate,2),round(mutstep,2)))
    
    best = individual()
    best.inherit(np.full_like(np.arange(n), upperLimit,dtype=np.double))

    for i in range(0,3):
        for j in range(0,3):
            test = population()
            
            mutrate = modification[k][1]
            mutstep = modification[k][2]
            print("Mutrate = {}%".format(mutrate))
            print("Mutstep = {}".format(mutstep))

            bestArray = []
            worstArray = []
            avrgArray = []
            genFitArray = []

            bestArray.append(test.best.fitness)
            worstArray.append(test.worst.fitness)
            avrgArray.append(test.average)

            for l in range(1,g):
                test.newGen()

                bestArray.append(test.best.fitness)
                worstArray.append(test.worst.fitness)
                avrgArray.append(test.average)

            if best.fitness > test.best.fitness:
                best = test.best

            print("Total mutations occured = ", test.mutCount)
 
            print("Final average fitness", myformat(test.average))
            print("Final Best Fitness:", myformat(test.best.fitness))
            ax[i,j].plot(avrgArray, linewidth=2.0, label='Average')
            ax[i,j].plot(bestArray, linewidth=2.0, label='Best')
            ax[i,j]. legend()
            ax[i,j].set_xlabel("Mutations = {}\nAverage = {}\n Best = {}".format(test.mutCount,myformat(test.average),myformat(test.best.fitness)))
            ax[i,j].xaxis.set_label_position('top')
            print("----------------------------------------------------------------------") 
 
    print("Test ", k+1, " results:")
    print(round(mutrate,2), round(mutstep,2), myformat(best.fitness))

    fig.savefig("further{}.png".format(k))
    plt.show()
 

 


