%**************************************************************************
%Script Classificador K-Nearest Neighbors *********************************
%**************************************************************************

function knn(path_arquivo)
%summary: Algoritmo K-Nearest Neighbors

clear all;
clc;
disp('-------------------------------------------------------------------');
disp('----------------- K-Nearest Neighbors -----------------------------');
disp('-------------------------------------------------------------------');

%--------------------------------------------------------------------------
% Carrega os dados --------------------------------------------------------
%--------------------------------------------------------------------------

global TrainingSet numTrainingSet;
global TestSet numTestSet classByTest Categories;
global k numDiscr numClass indCategory class_counts path_data  Y_Training;




format long;

if(nargin==0)
    %path_data = 'sampleWITHOUTINTMMEDIAGAMES.txt';
    path_data ='TRACES01To10WITHOUTINTMMEDIAGAMES.10Feats.txt';
    %path_data = 'entry09WITHOUTINTMMEDIAGAMES.10Features.txt';
else
    path_data = path_arquivo;
end
% Carrega os dados
[TrainingSet,TestSet] = LoadData(path_data,20);


% Numero de Instancias de Treinamento
[numTrainingSet,numTrainingSetColumns] = size(TrainingSet);
[numTestSet,numTestSetColumns] = size(TestSet);

%--------------------------------------------------------------------------
% Parametros de Configuracao ----------------------------------------------
%--------------------------------------------------------------------------


%k = floor(sqrt(numTestSet)); % Numero de vizinhos para comparar
k=1;

indCategory = numTrainingSetColumns;
numDiscr = numTrainingSetColumns - 1;

% Classes
Categories = ['WWW ';'MAIL';'FTP ';'ATCK';'P2P ';'DATB';'SERV';'MMED';'INTR';'GMES'];

% Numero de Classes Categorizadas
numClass = length(Categories) - 3;

% Vetor com as classes correspondentes aos dados de treinamento
Y_Training = TrainingSet(:,indCategory);

% Calcula contadores de classes
class_counts = zeros(1,numClass); 
for(indTrainingFlow=1:1:numTrainingSet)
    classFlow =  TrainingSet(indTrainingFlow,indCategory);
    class_counts(classFlow) = class_counts(classFlow) + 1;   
end


%--------------------------------------------------------------------------
% KNN ---------------------------------------------------------------------
%--------------------------------------------------------------------------
disp('classificando..');

classByTest = [zeros(1,numTestSet)]';
%Para cada fluxo nao classificado
for(indFlowTest=1:1:numTestSet)
    clc;
    disp(sprintf('fluxo de teste:%d de %d exemplos de teste',indFlowTest,numTestSet));
    classByTest(indFlowTest) = knn_classify(indFlowTest);    
end

Relatorio('K-Nearest Neighbors');
disp('terminado!');
%--------------------------------------------------------------------------
% Funçoes auxiliares ------------------------------------------------------
%--------------------------------------------------------------------------

function class = knn_classify(indFlowTest)
    [neighbors,dist_neighbors] = getMostNearNeighbors(indFlowTest); 
    class =  mostFrequentClass(neighbors,dist_neighbors);

function [neighbors,dist_neighbors] = getMostNearNeighbors(indFlowTest)
    global numTrainingSet k  numDiscr TrainingSet TestSet;
    %@summary: Obtem os k vizinhos mais proximos
    neighbors = [];
    dist_neighbors = [];
    %Obtem as distancias a cada ponto de treinamento
    dists = som_eucdist2(TrainingSet(:,1:numDiscr),TestSet(indFlowTest,1:numDiscr));
    %Seleciona os k pontos com menores distancias
    for(indFlow=1:1:k)
        index_minor = find(dists==min(dists));
        num_index_minor = length(index_minor);
        neighbors = [neighbors index_minor];
        dist_neighbors = [dist_neighbors dists(index_minor)];
        dists(index_minor) = Inf;
    end
    if(length(neighbors)>k)
        %neighbors = neighbors(1:k);
        %dist_neighbors = dist_neighbors(1:k);
    end
    clear dists;
    
function class = mostFrequentClass(neighbors,dist_neighbors)
    %@summary: Obtem classe mais frequente
    global Y_Training numClass indCategory k class_counts;
    frequencies = zeros(1,numClass);
    for(indNeighbor=1:1:k)
        c = Y_Training(neighbors(indNeighbor));
        frequencies(c) = frequencies(c) + 1; 
    end
    frequencies = frequencies/k;
    ind_max_freq = find(frequencies==max(frequencies));
    class = ind_max_freq(length(ind_max_freq)); 
    
    
function d = euclidian_dist(indFlowTest,indFlowTraining)
    global TestSet TrainingSet numDiscr;
    d = pdist( [ TestSet(indFlowTest,1:numDiscr) ; TrainingSet(indFlowTraining,1:numDiscr) ] , 'euclidean' );
    
    
function [Flows,Test] = LoadData(DataPath,PercentTest) 
  clc; 
  disp('Carregando..');
  DataFlows = load(DataPath);
  [numDataFlows,numDataColumns] = size(DataFlows);
  numDiscr = numDataColumns - 1;
  means_attr = mean(DataFlows(:,1:numDiscr));
  std_attr = std(DataFlows(:,1:numDiscr));  
  indCategory = numDataColumns;
  
  disp('Normaliza os dados');
  for(indFlow=1:1:numDataFlows)
    classFlow =  DataFlows(indFlow,indCategory); 
    for(indDiscr=1:1:numDiscr)
       DataFlows(indFlow,indDiscr) = (DataFlows(indFlow,indDiscr) -  means_attr(indDiscr))/std_attr(indDiscr);
    end
  end
  
  disp('Divide em dados de Treinamento e Teste'); 
  numFlowsTraining = floor(numDataFlows*(100-PercentTest)/100);
  numFlowsCategorise = numDataFlows - numFlowsTraining;
  Flows = DataFlows(1:numFlowsTraining,:);
  Test = DataFlows((numFlowsTraining+1):numDataFlows,:);
  if(numFlowsTraining==0)
    Flows = Test;
  end
  DataFlows = NaN;
  disp('->traces carregados.')    
  
  function Relatorio(scheme_name)
   global k numDiscr numClass TestSet numTestSet classByTest indCategory numTrainingSet Categories path_data
   
   fid = fopen('C:\\KNN_OUTPUT.txt','w');
   
   %Medicao da Acuracia da Classificacao (Exatidao Media)
   acertos = 0;
   num_fp = zeros(numClass,numClass); 
   num_fn = zeros(numClass,numClass);  
   cont_classes = zeros(1,numClass);
   acertos_por_classes =  zeros(1,numClass);
   for(indFlowTest=1:1:numTestSet)
       classFlowTest = TestSet(indFlowTest,indCategory);
       cont_classes(classFlowTest) = cont_classes(classFlowTest)  + 1;
       if(classByTest(indFlowTest) == classFlowTest)
           acertos=acertos+1;
           acertos_por_classes(classFlowTest) = acertos_por_classes(classFlowTest)  + 1;
       else
           %num da app sob analise classificada como outra app
           num_fn(classFlowTest,classByTest(indFlowTest)) = num_fn(classFlowTest,classByTest(indFlowTest)) + 1;
       end    
   end    
   
   % Exibe Informaçoes sobre o schema
   
   disp(sprintf('Scheme: %s',scheme_name(:)));
   fprintf(fid,'Scheme: %s\n',scheme_name(:));
   
   disp(sprintf('K-Nearest Neighbors = %d',k));
   fprintf(fid,'K-Nearest Neighbors = %d\n',k);
   
   disp(sprintf('Base de Dados: %s',path_data(:)));
   fprintf(fid,'Base de Dados: %s\n',path_data(:));
   
   disp(sprintf('Training Instances:%d',numTrainingSet));
   fprintf(fid,'Training Instances:%d\n',numTrainingSet);
   
   disp(sprintf('Test Instances:%d',numTestSet));
   fprintf(fid,'Test Instances:%d\n',numTestSet);

   % Exibe a Exatidao Media
   disp(sprintf('Qtde Acertos:%d',acertos));
   fprintf(fid,'Qtde Acertos:%d\n',acertos);
   
   disp(sprintf('Exatidao Media:%4.5f %%',acertos*100/numTestSet));
   fprintf(fid,'Exatidao Media:%4.5f %%\n',acertos*100/numTestSet);
   
   % Calcula matriz de confusao
   confusion_matrix = zeros(numClass,numClass);
   for(indClass_a=1:1:numClass)
       for(indClass_b=1:1:numClass)
           if(indClass_a==indClass_b)
               confusion_matrix(indClass_a,indClass_b) = acertos_por_classes(indClass_a);
           end
       end 
   end
   
   confusion_matrix = confusion_matrix + num_fn;
   
   disp('Matriz de Confusao:');
   fprintf(fid,'\nMatriz de Confusao:\n');
   disp(confusion_matrix);
   [ln,cl] = size(confusion_matrix);
   for(i=1:ln)
       fprintf(fid,'\n');
       for(j=1:cl)
           fprintf(fid,'\t%d',confusion_matrix(i,j));
       end
   end
   fprintf(fid,'\n');
   fclose(fid);
   