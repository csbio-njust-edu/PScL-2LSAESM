clear;
clc;
load all_features_DNA_Har;
load all_features_LBP;
load CLBP_all_feat;
load RICLBP_all_feat;
load LET_all_feat;
load all_labels;

DNA = DNA_Har(:,2:5);
Har = DNA_Har(:,6:end);
DNA(isnan(DNA))=0;
Har(isnan(Har))=0;
LBP(isnan(LBP))=0;
CLBP(isnan(CLBP))=0;
RICLBP(isnan(RICLBP))=0;
LET(isnan(LET))=0;

decValues = cell(1,10);
predectedLabels = cell(1,10);
predL = [];
actualL = [];
allProb = [];


CVO = cvpartition(label_all', 'k', 10);
for fold=1:CVO.NumTestSets
    trIdxUn=[]; teIdxUn=[]; DNAtr=[]; DNAte=[]; SLFstr=[]; SLFste=[];
    LBPtr=[]; LBPte=[]; CLBPtr=[]; CLBPte=[]; trainlabel=[]; testlabel=[]; 
    TR_Set = []; TE_Set = [];
    
    trIdxUn = CVO.training(fold);
    teIdxUn = CVO.test(fold);
    DNAtr= DNA(trIdxUn,:);
    DNAte = DNA(teIdxUn,:);
    Hartr = Har(trIdxUn,:);
    Harte = Har(teIdxUn,:);
    LBPtr = LBP(trIdxUn,:);
    LBPte = LBP(teIdxUn,:);
    CLBPtr = CLBP(trIdxUn,:);
    CLBPte = CLBP(teIdxUn,:);
    RICLBPtr = RICLBP(trIdxUn,:);
    RICLBPte = RICLBP(teIdxUn,:);
    LETtr = LET(trIdxUn,:);
    LETte = LET(teIdxUn,:);
    
    trainlabel= label_all(trIdxUn,:);
    testlabel =label_all(teIdxUn,:);
    
    %% Feature Normalization
    % SLfs Normalizatoin
    SLFstr = [DNAtr,Hartr];
    SLFste = [DNAte,Harte];
    [SLFstr, SLFste] = normalization(SLFstr, SLFste);
    % LBP Normalization
    [LBPtr, LBPte] = normalization(LBPtr, LBPte);
    % CLBP Normalizatoin
    [CLBPtr, CLBPte] = normalization(CLBPtr, CLBPte);
    % RICLBP Normalizatoin
    [RICLBPtr, RICLBPte] = normalization(RICLBPtr, RICLBPte);
    % LET Normalizatoin
    [LETtr, LETte] = normalization(LETtr, LETte);
   
    %% Feature Selection
    % SLfs Feature Selection
    [idx_sdaSLFs SLFstr SLFste] = SDA_FeatSelect(double(SLFstr), double(SLFste), trainlabel);
    % LBP Feature Selection
    [idx_sdaLBP LBPtr LBPte] = SDA_FeatSelect(double(LBPtr), double(LBPte), trainlabel);
    % CLBP Feature Selection
    [idx_sdaCLBP CLBPtr CLBPte] = SDA_FeatSelect(double(CLBPtr), double(CLBPte), trainlabel);
    % RICLBP Feature Selection
    [idx_sdaRIC RICLBPtr RICLBPte] = SDA_FeatSelect(double(RICLBPtr), double(RICLBPte), trainlabel);
    % LET Feature Selection
    [idx_sdaLET LETtr LETte] = SDA_FeatSelect(double(LETtr), double(LETte), trainlabel);
    
    %% Get 1st-Level trained SAE-SMs and predict
    [Level2_SLFste,predL1,acc_softmax1] = L1_SAE_SM(SLFstr, trainlabel, SLFste, testlabel);
    [Level2_LBPte,predL2,acc_softmax2] = L1_SAE_SM(LBPtr, trainlabel, LBPte, testlabel);
    [Level2_CLBPte,predL3,acc_softmax3] = L1_SAE_SM(CLBPtr, trainlabel, CLBPte, testlabel);
    [Level2_RICLBPte,predL4,acc_softmax4] = L1_SAE_SM(RICLBPtr, trainlabel, RICLBPte, testlabel);
    [Level2_LETte,predL5,acc_softmax5] = L1_SAE_SM(LETtr, trainlabel, LETte, testlabel);
    
    %% Get SAE-SM* and 2nd-Level Train Set
    
    nestedCVO = cvpartition(trainlabel', 'k', 10);
    nestedtrIdx=[];
    nestedteIdx=[];
    for nestedfold=1:nestedCVO.NumTestSets
           nestedtrIdx{nestedfold} = nestedCVO.training(nestedfold);
           nestedteIdx{nestedfold} = nestedCVO.test(nestedfold);
    end

    [level2_SLFstr, trainY] = get_L2_trD(SLFstr, trainlabel, nestedtrIdx, nestedteIdx);
    
    [level2_LBPtr, trainYl] = get_L2_trD(LBPtr, trainlabel, nestedtrIdx, nestedteIdx);
    
    [level2_CLBPtr, trainY2] = get_L2_trD(CLBPtr, trainlabel, nestedtrIdx, nestedteIdx);
    
    [level2_RICLBPtr, trainY3] = get_L2_trD(RICLBPtr, trainlabel, nestedtrIdx, nestedteIdx);
    
    [level2_LETtr, trainY4] = get_L2_trD(LETtr, trainlabel, nestedtrIdx, nestedteIdx);
    
    
    %% Integrating features via mean ensemble method
    TR_Set = (level2_SLFstr+level2_LBPtr+level2_CLBPtr+level2_RICLBPtr+level2_LETtr)./5;
    TE_Set = (Level2_SLFste+Level2_LBPte+Level2_CLBPte+Level2_RICLBPte+Level2_LETte)./5;

%     [TR_Set, TE_Set] = normalization(TR_Set, TE_Set);
    
    %% Training the 2nd-Level SAE-SM
    [FinalTEprob,FinalPreLabelTe,acc_Final_softmax] = L2_SAE_SM(double(TR_Set),trainY,double(TE_Set),testlabel);
    
    twoL_SAESMAcc(fold) = acc_Final_softmax;
    allProb = [allProb;FinalTEprob];
    decValues(fold) = {FinalTEprob};
    predectedLabels(fold)= {FinalPreLabelTe};
    predL = [predL;FinalPreLabelTe];
    actualL = [actualL;testlabel];
end

Final_Acc_SAESM = mean(twoL_SAESMAcc)

twoL_SAESM.Acc = twoL_SAESMAcc;
twoL_SAESM.decValues = decValues;
twoL_SAESM.allProb = allProb;
twoL_SAESM.predectedLabels = predectedLabels;
twoL_SAESM.predL = predL;
twoL_SAESM.actualL = actualL;
save SAESM_Final_Results twoL_SAESM;