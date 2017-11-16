

function zeroR(path_arquivo)


%--------------------------------------------------------------------------
% Parametros de Configuracao ----------------------------------------------
%--------------------------------------------------------------------------

% Variaveis globais
global numDiscr numClass numFlows counts means desv priors Flows numFlows Test numTest dctFluxosClasses indCategory 
global Categories startIndexes flowsbyclass indexflowsbyclass path_data max_freq_class;    
format long;
if(nargin==0)
    path_data = 'C:\CLASSIFIERS\NaiveBayesClassifier\samples\sampleWITHOUTINTMMEDIAGAMES.txt';
else
    path_data = path_arquivo;
end
%path_data = 'C:\CLASSIFIERS\DATA\TXT\10\TRACES01To10WITHOUTINTMMEDIAGAMES.10Features.txt';
% Carrega os dados
[Flows,Test] = LoadData(path_data,20);


% Numero de Instancias de Treinamento
[numFlows,numColumns] = size(Flows);
[numTest,numColumns] = size(Test);



% Classes
Categories = ['WWW ';'MAIL';'FTP ';'ATCK';'P2P ';'DATB';'SERV';'MMED';'INTR';'GMES'];

% Numero de Classes Categorizadas
numClass = length(Categories) - 3;

% Numero de Discriminantes
numDiscr = numColumns - 1;
% Indice da coluna CLASSE
indCategory = numColumns;


%-----------------------------------------------
% TREINAMENTO
%-----------------------------------------------

numClassByflows = zeros(1,numClass); 
for(indFlow=1:1:numFlows)
    classFlow =  Flows(indFlow,indCategory);
    numClassByflows(classFlow) = numClassByflows(classFlow) + 1;   
end


max_freq_class = find(numClassByflows==max(numClassByflows));
max_freq_class = max_freq_class(length(max_freq_class));
%-----------------------------------------------
% CATEGORIZACAO
%-----------------------------------------------

disp('Categorizando ..');
dctFluxosClasses = [zeros(1,numTest)]';
for(indFlowTest=1:1:numTest)
    clc;
    disp(sprintf('%4.5f %% concluido.',(indFlowTest*100)/numTest));
    dctFluxosClasses(indFlowTest) = max_freq_class;
end



%-----------------------------------------------
% RELATORIO
%-----------------------------------------------
Relatorio('Naive Bayes (Simple)');
disp('done.');



%-----------------------------------------------
% FUNÇOES
%-----------------------------------------------

function Relatorio(scheme_name)
   global numDiscr numClass Test numTest dctFluxosClasses indCategory numFlows Categories means desv priors path_data
   
   fid = fopen('C:\NBSIMPLE_OUTPUT.txt','w');
   
   %Medicao da Acuracia da Classificacao (Exatidao Media)
   acertos = 0;
   num_fp = zeros(numClass,numClass); 
   num_fn = zeros(numClass,numClass);  
   cont_classes = zeros(1,numClass);
   acertos_por_classes =  zeros(1,numClass);
   for(indFlowTest=1:1:numTest)
       classFlowTest = Test(indFlowTest,indCategory);
       cont_classes(classFlowTest) = cont_classes(classFlowTest)  + 1;
       if(dctFluxosClasses(indFlowTest) == classFlowTest)
           acertos=acertos+1;
           acertos_por_classes(classFlowTest) = acertos_por_classes(classFlowTest)  + 1;
       else
           lalala = 0;
           %num da app sob analise classificada como outra app
           num_fn(classFlowTest,dctFluxosClasses(indFlowTest)) = num_fn(classFlowTest,dctFluxosClasses(indFlowTest)) + 1;
       end    
   end    
   Exatidao_Media_Por_App = acertos_por_classes./cont_classes;
   
   % Exibe Informaçoes sobre o schema
   
   disp(sprintf('Scheme: %s',scheme_name(:)));
   fprintf(fid,'Scheme: %s\n',scheme_name(:));
   
   disp(sprintf('Base de Dados: %s',path_data(:)));
   fprintf(fid,'Base de Dados: %s\n',path_data(:));
   
   disp(sprintf('Training Instances:%d',numFlows));
   fprintf(fid,'Training Instances:%d\n',numFlows);
   
   disp(sprintf('Test Instances:%d',numTest));
   fprintf(fid,'Test Instances:%d\n',numTest);
   
   % Exibe probabilidades,medias e desvios padrao dos atributos
   for(indClass=1:1:numClass)
       
       disp(sprintf('ProbClasse(%s) = %4.5f',Categories(indClass,:),priors(indClass)));
       fprintf(fid,'ProbClasse(%s) = %4.5f\n',Categories(indClass,:),priors(indClass));
       
       for(indDiscr=1:1:numDiscr)
           rMean = sprintf('mean(%d| %s) = %15.5f',indDiscr,Categories(indClass,:),means(indClass,indDiscr));
           rDesv = sprintf('desv(%d| %s) = %15.5f',indDiscr,Categories(indClass,:),desv(indClass,indDiscr));
           
           disp(sprintf('%s %s',rMean,rDesv));
           fprintf(fid,'%s %s\n',rMean,rDesv);
           
       end
   end
   % Exibe a Exatidao Media
   disp(sprintf('Qtde Acertos:%d',acertos));
   fprintf(fid,'Qtde Acertos:%d\n',acertos);
   
   disp(sprintf('Exatidao Media:%4.5f %%',acertos*100/numTest));
   fprintf(fid,'Exatidao Media:%4.5f %%\n',acertos*100/numTest);
   
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
   
function prob = Normal(x,u,o)
    %@summary: pdf Normal
    %@param x: valor numerico
    %@param u: media
    %@param o: desvio padrao 
    prob = (1/(sqrt(2*pi)*o)*exp((-(x-u)^2)/(2*o^2)));

function ProdD = ProbDataGivenClass(indFlowTest,indClass)
        
        global numDiscr means desv Test flowsbyclass indexflowsbyclass Flows numFlows indCategory counts startIndexes;
        
        ProdD = 1;
        for(indDiscr=1:1:numDiscr)
            u = means(indClass,indDiscr);
            o = desv(indClass,indDiscr);
            x = Test(indFlowTest,indDiscr);
            ProdD = ProdD * Normal(x,u,o);     
        end
        

        
 function elected_class = arg_max(indFlowTest)
 
    global numClass priors
    
    elected_class = -1;
    elected_prob  = -1;
    for(indClass=1:1:numClass)
        ProbC = priors(indClass)*ProbDataGivenClass(indFlowTest,indClass);
        if ProbC > elected_prob
             elected_class = indClass;
             elected_prob = ProbC;
        end  
     end
        
        
function k = gaussianDistr(t)
        k = (1/(sqrt(2*pi))) * exp((-(t^2))/2)
        

function [Flows,Test] = LoadData(DataPath,PercentTest) 
  clc; 
  disp('Carregando..');
  DataFlows = load(DataPath);
  [numDataFlows,numDataColumns] = size(DataFlows);
  numFlowsTraining = floor(numDataFlows*(100-PercentTest)/100);
  numFlowsCategorise = numDataFlows - numFlowsTraining;
  Flows = DataFlows(1:numFlowsTraining,:);
  Test = DataFlows((numFlowsTraining+1):numDataFlows,:);
  if(numFlowsTraining==0)
    Flows = Test;
  end
  DataFlows = NaN;
  disp('->traces carregados.')
  




