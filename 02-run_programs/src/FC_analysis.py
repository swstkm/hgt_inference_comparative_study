''' This script does all the functional category analysis. It includes the calculation of absolute and relative transferability, fisher exact test on relative transferability, Kruskal-Wallis test on absolute transferability and Mann-Whitney test on meta categories. '''

#list of python libraries
import os
import ete3
import argparse
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy.stats as stats


parser = argparse.ArgumentParser()
parser.add_argument('-m',metavar='memberfile',type=str,help='eggNOG memberfile')
parser.add_argument('-a',metavar='annotations',type=str,help='eggNOG annotations')
parser.add_argument('-i',metavar='HGT file',type=str,help='HGT outfile. Each line should contain only one putative HGT candidate.')
parser.add_argument('-g',metavar='Input type',type=int,help='HGT input type. Wn_threshold returns a file with putative HGT candidates, but with specific gene IDs, while phylogenetic programs return either pairwise recipient donor pairs or only recipients, both as taxonomic IDs. Specify the input as 0 -> gene, 1 -> taxonomix ID')
parser.add_argument('-o',metavar='outfile name',type=str,help='Outfile name with pathing.')

parser.add_argument('-f',metavar='BH correction, false positive rate',type=int,default=50,help='BH correction parameter to set the ratio of false positives.')
args = parser.parse_args()

if not args.i:
								parser.error("No HGT input given. Please check your inputs again.")
								exit
if not args.o:
								parser.error("No outfil name. Please check your inputs again.")
								exit
if not args.m:
								parser.error("No memberfile to be found. Please check your inputs again.")
								exit
if not args.a:
								parser.error("No annotation file to be found. Please check your inputs again.")
								exit
#############################################################################
NOG2IDs = dict() # NOG to taxonomic IDs
NOG2HGT = dict() # NOG to the amount of HGT
prot2NOGs = dict() #protein acession ID to its NOG
ID2NOGs = dict()
NOG2FC = dict() # NOG to its functional category
FC2HGT= dict() #Functional category to a list of proteins belonging to the functional category

cats = ["C","G","E","F","H","I","P","Q","V","D","T","M","N","U","O","J","K","L"] 
cat_to_total = dict() # counts the amount of NOGs of one functional category
cat_to_transfers = dict() # counts the amount of transfers/gains of one functional category
cat_to_transferable = dict() # counts the amount of transferable NOGs of one functional category
cat_to_description = {
								"C":"Energy production and conversion",
								"G":"Carbohydrate transport and metabolism",
								"E":"Amino acid transport and metabolism",
								"F":"Nucleotide transport and metabolism",
								"H":"Coenzyme transport and metabolism",
								"I":"Lipid transport and metabolism",
								"P":"Inorganic ion transport and metabolism",
								"Q":"Secondary metabolites biosynthesis, transport and catabolism",
								"V":"Defense mechanisms",
								"D":"Cell cycle control, cell division, chromosome partitioning",
								"T":"Signal transduction mechanisms",
								"M":"Cell wall/membrane/envelope biogenesis",
								"N":"Cell motility",
								"U":"Intracellular tracking, secretion, and vesicular transport",
								"O":"Post-translational modi cation, protein turnover, chaperones",
								"J":"Translation, ribosomal structure and biogenesis",
								"K":"Transcription",
								"L":"Replication, recombination and repair"
}

meta2tottrans = dict()
meta_to_total = dict() ## counts the amount of gene families of one meta category
meta_to_transfers = dict() ## counts the amount of transfers/gains of one meta category
meta_to_transferable = dict() ## counts the amount of transferable gene families of one meta category

FC2tottrans = dict() #FC to all transfers in a list; each element of the list contains the amount of transfers of one NOG; needed for the Kruskal-Wallis test
meta2tottrans = dict() #same as above, only with meta categories

## Declare a dictionary where each functional category points to their corresponding meta category.
## A -> metabolic function, B -> cellular processes/signaling, C -> informational
cat_to_meta = {
        "C":"A",
        "G":"A",
        "E":"A",
        "F":"A",
        "H":"A",
        "I":"A",
        "P":"A",
        "Q":"A",
        "V":"B",
        "D":"B",
        "T":"B",
        "M":"B",
        "N":"B",
        "U":"B",
        "O":"B",
        "J":"C",
        "K":"C",
        "L":"C"
    } 

# BH-correction function
def BH(pvalues):
								pvalues = dict(sorted(pvalues.items(), key=lambda x:x[1]))
								FCs = list(pvalues.keys())
								critical = 0
								pvalue2significant = dict()
								for i in range(len(FCs)):
																pvalue = pvalues[FCs[i]]
																correction = (i+1/len(FCs))*(args.f/100)
																if pvalue < correction:
																								if pvalue > critical:
																																critical = pvalue
																
																pvalue2significant[FCs[i]] = "no"
								for i in range(len(FCs)):
																if pvalues[FCs[i]] < critical:
																								pvalue2significant[FCs[i]] = "yes"
								return pvalue2significant
################################################################################
for line in open(args.m):
								line = line.rstrip()
								line = line.split("\t")
								NOG = line[1]
								protnames = line[4].split(",") # list containing all protein IDs
								taxIDs = line[5].split(",") # list containing all taxonomic IDs
								NOG2IDs[NOG] = taxIDs
								NOG2HGT[NOG] = list()
								for name in protnames:
																if name not in prot2NOGs.keys():
																								prot2NOGs[name] = NOG
																else:
																								prot2NOGs[name] += ","+NOG
								for ID in taxIDs:
																if ID not in ID2NOGs.keys():
																								ID2NOGs[ID] = NOG
																else:
																								ID2NOGs[ID] += ","+NOG
																								
for line in open(args.a):
        line = line.rstrip()
        line = line.split("\t")
        NOG = line[1]
        FC = line[2]
        NOG2FC[NOG] = FC
        if FC not in FC2HGTkeys():
                FC2HGT[FC] = list()
                
# depending on the HGT input (either gene IDs or taxonomic IDs), the general procedure differs by a bit. 
if args.g == 0: # if the HGT input contains gene IDs
								for line in open(args.i):
																HGTs = line.rstrip().split(",")
																for HGT in HGTs:
																								if HGT in prot2NOGs.keys():
																																for NOG in prot2NOGs[HGT].split(","):
																																								NOG2HGT[NOG].append(HGT)

else: # if the HGT input contains taxnomic IDs
								for line in open(args.i):
																HGTs = line.rstrip().split(",")
																for HGT in HGTs:
																								if HGT in ID2NOGs.keys():
																																for NOG in ID2NOGs[HGT].split(","):
																																								NOG2HGT[NOG].append(HGT)
 #########################################################################################################                                                 
# Now sort the NOGs into their functional category. After that, count the amount of HGT per functional category/meta category    


for NOG in NOG2HGT:
        if NOG in NOG2FC.keys():
                HGTs = NOG2HGT[NOG]
                FC = NOG2FC[NOG]
                for HGT in HGTs:
                        FC2HGT[FC].append(HGT)
                        
''' calculating the relative and absolute transferability for all functional and meta categories '''
#### All FC
rates_FC = open(args.o+"_rates_FC.txt","w") #this is the outfile

for cat in cats: ## We are declaring every key-value pair in the three dictionaries above. Key will be the functional category and the value will be 0.
        cat_to_total[cat] = 0
        cat_to_transfers[cat] = 0
        cat_to_transferable[cat] = 0
        
for NOG in NOG2FC:
        
        NOG = NOG
        cat = NOG2FC[NOG]
        trans = len(NOG2HGT[NOG])
        for i in range(len(cat)): ## There might be gene families with more than one functional category, so we iterate through the whole length of "cat".
                onecat = cat[i]
                if onecat in cats: ## Check if this functional category is even in the list of functional categories we are looking for
                        cat_to_total[onecat] += 1
                        cat_to_transfers[onecat] += trans
                        if cat not in FC2tottrans.keys():
                                FC2tottrans[cat] = list()
                                FC2tottrans[cat].append(trans)
                        else:
                                FC2tottrans[cat].append(trans)
                        if trans > 1:
                                cat_to_transferable[onecat] += 1

## Now calculate all rates.
sum_total = sum(list(cat_to_total[FC] for FC in cat_to_total.keys()))
sum_transferables = sum(list(cat_to_transferable[FC] for FC in cat_to_transferable.keys())) 
rates_FC.write("#FC\tabtrans\treltrans\n")
for cat in cats:
        transfers = cat_to_transfers[cat]
        total = cat_to_total[cat]
        transferable = cat_to_transferable[cat]
        nontransferable = total - transferable

        abtrans = transfers/total
        rate = transferable/total
        reltrans = rate/(sum_transferables/sum_total)
        rates_FC.write(cat+"\t"+str(abtrans)+"\t"+str(reltrans)+"\n")
rates_FC.close()

#### Meta categories
rates_meta = open(args.o+"_rates_meta.txt","w")

#later needed for Fisher exact test
total_transfers = 0
total_prots = 0
total_transferable = 0

for meta in ["A","B","C"]: ## We are declaring every key-value pair in the three dictionaries above. Key will be the meta category and the value will be 0.
        meta2tottrans[meta] = list()
        meta_to_total[meta] = 0
        meta_to_transfers[meta] = 0
        meta_to_transferable[meta] = 0
        
for NOG in NOG2FC:
        
        NOG = NOG
        cat = NOG2FC[NOG]
        trans = len(NOG2HGT[NOG])
        
        for i in range(len(cat)): ## There might be gene families with more than one functional category, so we iterate through the whole length of "cat".
                onecat = cat[i]
                if onecat in cats: ## Check if this functional category is even in the list of functional categories we are looking for
                        ## get the meta category of the current functional category
                        meta = cat_to_meta[onecat]
                
                        meta_to_total[meta] += 1
                        total_prots += 1  #Fisher
                        meta_to_transfers[meta] += trans
                        total_transfers += trans  #Fisher
                        meta2tottrans[meta].append(trans)
                        if float(trans) > 2:
                                meta_to_transferable[meta] += 1
                                total_transferable += 1  #Fisher
                                
                                
## Now calculate all transferability values.  
sum_total = sum(list(meta_to_total[meta] for meta in meta_to_total.keys()))
sum_transferables = sum(list(meta_to_transferable[meta] for meta in meta_to_transferable.keys())) 
rates_meta.write("#meta\tabtrans\treltrans\n")
category_name = {
        "A":"metabolic function",
        "B":"cellular processes/signalling",
        "C":"informational"
}
for meta in ["A","B","C"]:
        transfers = meta_to_transfers[meta]
        total = meta_to_total[meta]
        transferable = meta_to_transferable[meta]
        nontransferable = total - transferable

        abtrans_meta = transfers/total
        rate_meta = transferable/total
        reltrans_meta = rate_meta/(sum_transferables/sum_total)
        rates_meta.write(meta+"\t"+str(abtrans_meta)+"\t"+str(reltrans_meta)+"\n")
rates_meta.close()


''' Fisher exact test on relative transferability '''
#For all the FC we have the three dictionaries to use:
#cat_to_total
#cat_to_transfers
#cat_to_transferable
fisher_outfile = open(args.o+"_fisher_allFC.txt","w")
total_transferable_FC = sum(list(cat_to_transferable[FC] for FC in cat_to_transferable.keys()))

pvalues = dict()
oddsratios = dict()
for FC in cat_to_transferable.keys():
								transferable = cat_to_transferable[FC]
								nontransferable = cat_to_total[FC] - transferable
								cont_transferable = total_transferable_FC - transferable
								cont_nontransferable = sum(list(cat_to_total[FC] for FC in cat_to_total.keys())) - total_transferable_FC - nontransferable

								data = [[transferable,nontransferable],[cont_transferable,cont_nontransferable]]
								df = pd.DataFrame(data)
								oddsratio,pvalue = stats.fisher_exact(df)
								pvalues[FC] = pvalue
								oddsratios[FC] = oddsratio
								
BH_pvalues = BH(pvalues)
for FC in cats:
								pvalue = pvalues[FC]
								significant = BH_pvalues[FC]
								oddsratio = oddsratios[FC]
								if significant == "yes":
																fisher_outfile.write(cat_to_description[FC]+" ("+FC+")\t"+str(pvalue)+"\t"+str(oddsratio)+"\t*\n")
								else:
																fisher_outfile.write(FC+"\t"+str(pvalue)+"\t"+str(oddsratio)+"\t \n")
fisher_outfile.close()


#For the meta categories we have the three dictionaries to use:
#meta_to_total
#meta_to_transfers
#meta_to_transferable
pvalues = dict()
oddsratios = dict()
fisher_outfile = open(args.o+"_fisher_allmeta.txt","w")
total_transferable_meta = sum(list(meta_to_transferable[meta] for meta in ["A","B","C"]))
for meta in ["A","B","C"]:
								transferable = meta_to_transferable[meta]
								nontransferable = meta_to_total[meta] - transferable
								cont_transferable = total_transferable_meta - transferable #this is the total amount of transfers across all meta categories excluding the current meta cat
								cont_nontransferable = sum(list(meta_to_total[meta] for meta in meta_to_total.keys())) - total_transferable_meta - nontransferable #this is the total amount of non-transfers across all meta categories excluding the current meta cat
        
        #print(transferable,nontransferable,cont_transferable,cont_nontransferable)
								data = [[transferable,nontransferable],[cont_transferable,cont_nontransferable]]
								df = pd.DataFrame(data)
								oddsratio, pvalue = stats.fisher_exact(df)
								pvalues[meta] = pvalue
								oddsratios[meta] = oddsratio
								
BH_pvalues = BH(pvalues)
for meta in ["A","B","C"]:
								pvalue = pvalues[meta]
								significant = BH_pvalues[meta]
								oddsratio = oddsratios[meta]
								if significant == "yes":
																fisher_outfile.write(category_name[meta]+" ("+meta+")\t"+str(pvalue)+"\t"+str(oddsratio)+"\t*\n")
								else:
																fisher_outfile.write(meta+"\t"+str(pvalue)+"\t"+str(oddsratio)+"\t \n")
fisher_outfile.close()

''' Kruskal-Wallis-test on absolute transferability '''
kruskal_outfile = open(args.o+"_kruskal_results.txt","w")
#### for all FC
K = list()
for FC in FC2tottrans.keys():
        K.append(FC2tottrans[FC])

kruskal_outfile.write(str(stats.kruskal(K[0],K[1],K[2],K[3],K[4],K[5],K[6],K[7],K[8],K[9],K[10],K[11],K[12],K[13],K[14],K[15],K[16],K[17]))+"\n")
kruskal_outfile.close()

''' Mann-Whitney test on meta categories '''
MW_outfile = open(args.o+"_mannwhitney_results.txt","w")
M = list()
for meta in meta2tottrans:
        transf=meta2tottrans[meta]
        transf=[float(transf[j]) for j in range(1,len(transf))]
        M.append(transf)
        
AB = stats.mannwhitneyu(M[0],M[1])
AC = stats.mannwhitneyu(M[0],M[2])
BC = stats.mannwhitneyu(M[1],M[2])

MW_outfile.write(str(AB)+"\n")
MW_outfile.write(str(AC)+"\n")
MW_outfile.write(str(BC)+"\n")
MW_outfile.close()


