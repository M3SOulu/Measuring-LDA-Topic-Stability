#This is source code for the experiments done in paper "Measuring LDA Topic Stability from Clusters of Replicated Runs" by M. Mantyla, M. Claes, U. Farooq. 
#See Arxiv for the paper

#Cleanup and environment setup
rm(list = ls())
setwd("C:/Users/mmantyla/Dropbox/Research - Publications/201x - Explorations/2018 - Mozilla - POS tagger")
setwd("D:/Dropbox/Research - Publications/201x - Explorations/2018 - Mozilla - POS tagger")
data <- "D:/Dropbox/Data/Apache_Mozilla/"
#data <- "C:/Users/mmantyla/Dropbox/Data/Apache_Mozilla/"


#Read and Prepare Data-------------------------------------------------------------
library("data.table")
idmerging <- fread(file=paste(data,sep="","identity_merging.csv"), stringsAsFactors=FALSE)
gitlog_filtered <- readRDS(file=paste(data,sep="","git_filtered.rds"))
#gitlog_all <- readRDS(file=paste(data,sep="","git.rds"))
mozdev <- fread(file=paste(data,sep="","mozilla_developers.csv"))
mozdev <- read.csv(file=paste(data,sep="","mozilla_developers.csv"))

gitlog_filtered[, hired := merged.id %in% mozdev$merged.id]
#merges ids
gitlog_filtered <- gitlog_filtered[, .SD[1], by=c("source", "project", "hash")]
gitlog_filtered$added <- NULL
gitlog_filtered$removed <- NULL
gitlog_filtered$extension <- NULL

prepare_input <- function (){
  log_ff <- gitlog_filtered[project == "Core" | project=="Firefox"]
  log_ff[, office.hours := !weekend & hour >= 10 & hour < 18]
  log_ff[, rapid.release := date >= as.Date("2011-03-22")]
  #cut r= and a= reviers and assginees. 
  #Keep these removing them all is very hard. Remove by username instead. 
  #log_ff$message <- gsub("r=+[^(\\n|\\s)]*","",log_ff$message, perl=TRUE)
  #log_ff$message <- gsub("a=+[^(\\n|\\s)]*","",log_ff$message, perl=TRUE)
  
  log_ff$message <- gsub("Signed-off-by:+[^\\n]*","",log_ff$message, perl=TRUE)
  #remove servo standard check box commit contents
  log_ff$message <- gsub("<!--+[^(\\z)]*", "", log_ff$message, perl=TRUE)

  #Through manual inspection log message longer than that contain mostly non natural language
  #In firefox this removes 15 messages
  log_ff <- log_ff[nchar(log_ff$message) < 4850]
  #returnvalue log_ff;
}
log_ff <- prepare_input()

#Stopwords list, including list of userid names, is built here
prepare_stopwords <- function () {
  library("stringr")
  names_ff <- str_split_fixed (log_ff$author, "@", 2)
  names_ff <- str_split_fixed (names_ff, "<", 2)
  names_ff <- unique(names_ff[,2])
  #hand added userids
  names_ff <- c(names_ff, "smaug", "beltzner", "gmail.com", "tbsaunde", 
              "surkov", "vlad", "jandem", "bhackett", "bholley", "heycam",
              "bbouvier", "dvander", "jimm", "schrep", "mano", "mak", "bz",
              "keeler", "karlt", "mt", "gerv", "brendan", "bzbarsky", "jesse", 
              "ruderman", "philikon", "marcoz", "davidb", "cjones", "callek", 
              "jduell", "waldo", "bsmedberg", "gerald", "jya", "jesup", "bkelly",
              "ritu", "lizzard", "gchang", "gijs", "glandium", "xidorn", "baldr",
              "jonco") 
  stop_words <- read.csv("scikitlearn.txt", sep = "", stringsAsFactors=FALSE, header = FALSE)[,1]
  stop_words <- c(stop_words, read.csv("snowball_expanded.txt", sep = "", stringsAsFactors=FALSE, header = FALSE)[,1])
  stop_words <- c(stop_words, names_ff, "sr", "b", "r", "sr")
  stop_words <-  unique(stop_words)
}
stop_words <- prepare_stopwords()


#text clustering-------------------------------------------------------------
library("magrittr")
library("text2vec")
library("tokenizers")

#Find optimal hyper priors alpha and beta that are used later
optimalLda <- function (x){
  t1 = Sys.time()
  sink("NUL")
  m_k <- round (x[1])
  m_alpha <- x[2]
  m_beta <- x[3]
  
  sample <- sample.int(n = nrow(log_ff), size = floor(.80*nrow(log_ff)), replace = F)

  tokens = log_ff$message [sample] %>%  tokenize_words (strip_numeric = TRUE, stopwords = stop_words)
  it <- itoken(tokens, progressbar = FALSE)

  v = create_vocabulary(it) %>% 
    prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.3)
  vectorizer = vocab_vectorizer(v)

  dtm = create_dtm(it, vectorizer, type = "dgTMatrix")

#Find correct hyper parameters. 
  lda_model = LDA$new(n_topics = m_k, doc_topic_prior = m_alpha, topic_word_prior = m_beta)
  doc_topic_distr <- 
    lda_model$fit_transform(x = dtm, n_iter = 1000, 
                            #convergence_tol = 0.001, n_check_convergence = 25, 
                            convergence_tol = 0.01, n_check_convergence = 25, 
                            progressbar = FALSE, verbose=FALSE)#)
  
#apply to training set
  new_dtm = itoken(log_ff$message[-sample], tolower, word_tokenizer) %>% 
    create_dtm(vectorizer, type = "dgTMatrix")
  new_doc_topic_distr = lda_model$transform(new_dtm)
  sink()
  perp <- perplexity(new_dtm, topic_word_distribution = lda_model$topic_word_distribution, doc_topic_distribution = new_doc_topic_distr)
  m_k <- round (x[1])
  m_alpha <- x[2]
  m_beta <- x[3]
  print(paste("k:", m_k, "alpha:", m_alpha, "beta", m_beta, 
              "perp:", perp, "time used: ", difftime(Sys.time(), t1, units = 'sec')))
  #difftime(Sys.time(), t1, units = 'sec')
  perp
  
}
lower <- c(20, 0, 0)
higher <- c(20, 1.0, 0.1)
library(DEoptim)
#Perform the DE search of optimal parameters. 
#Warning!!! takes time and resulting alpha and beta are documented lower
DEoptim(optimalLda, lower, higher, DEoptim.control(strategy = 2, itermax = 10, NP = 20))

#Create document-term matrix for creating LDA models-----------------------------------------------------------
tokens = log_ff$message  %>%  tokenize_words (strip_numeric = TRUE, stopwords = stop_words)
it <- itoken(tokens, progressbar = FALSE)
vocab = create_vocabulary(it) %>% 
  prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.3)
vectorizer = vocab_vectorizer(vocab)
dtm = create_dtm(it, vectorizer, type = "dgTMatrix")

#Here the LDA model is created. n is number of replicated runs
#Paper: "When we cluster replicated LDA runs we have $n$ replicated runs, and each run contains $k$ number of topics."
create_lda <- function (n=3){

  tw_dist_store <<- NULL #variable to store topic-word matrix
  dt_dist_store <<- NULL #variable to store document-topic matrix
  tw_lift_dist_store <<- NULL #variable to store topic-word lift (from LDAVis paper) matrix. Not used in ESEM short but can be computed 
  #by running the the commented code in below
  #tw_dist_with_lift = exp(
  #  my_lambda * log(lda_model$topic_word_distribution) +
  #    (1 - my_lambda) * log(t(t(lda_model$topic_word_distribution) / 
  #                              (colSums(lda_model$components) / 
  #                                 sum(lda_model$components)) )))
  for (i in 1:n){
    lda_model = LDA$new(n_topics = 20, doc_topic_prior =  0.16727116, topic_word_prior = 0.07609007) #Alpha and Beta have been solved earlier and hard coded here
    doc_topic_distr = 
      lda_model$fit_transform(x = dtm, n_iter = 1000, 
                              convergence_tol = 0.0001, n_check_convergence = 100, 
                              progressbar = TRUE)
    tw_dist <- lda_model$topic_word_distribution
    tw_lift_dist <- t(t(lda_model$topic_word_distribution) / 
                           (colSums(lda_model$components) / 
                           sum(lda_model$components)) )
  
    tw_dist_store <<- rbind(tw_dist_store, tw_dist)
    tw_lift_dist_store <<- rbind(tw_lift_dist_store, tw_lift_dist)
    dt_dist_store <<- cbind(dt_dist_store, doc_topic_distr)
  }
}

#Load pre-computed data files from disk if not found create them. 
t1 = Sys.time()
if (!exists("tw_dist_store") || !exists("dt_dist_store") || !exists("tw_lift_dist_store")){
  result = tryCatch(
    {
      tw_dist_store <- readRDS("tw_dist_store.rds")
      tw_dist_store[is.na(tw_dist_store)] <- 0
      dt_dist_store <- readRDS("dt_dist_store.rds")
      tw_lift_dist_store <-  readRDS("tw_lift_dist_store.rds")
    },
    error=function(cond) {
      create_lda(n=20)#how many LDAs important to set
      tw_dist_store[is.na(tw_dist_store)] <- 0
      saveRDS(tw_dist_store, file= "tw_dist_store.rds")
      saveRDS(dt_dist_store, file= "dt_dist_store.rds")
      saveRDS(tw_lift_dist_store, file= "tw_lift_dist_store.rds")
    }
  )
} 
tw_dist_store[is.na(tw_dist_store)] <- 0
print(difftime(Sys.time(), t1, units = 'sec'))
#Glove clustering----------------------------------------------------------------------
#We use entire space for input as we search for synonyms. Glove is fast and more is better
#Create GLoVE vector space
#Paper: "Thus, we form a word vector space with $w$ words and $v$ vectors as matrix $V$ with dimensions ($w\times v$)."
create_glove <- function (){
  library (magrittr)
  library (text2vec)
  library(tokenizers)
  log_ff_g <- prepare_input()
  
  tokens = log_ff_g$message  %>%  tokenize_words (strip_numeric = TRUE, stopwords = stop_words)
  it <- itoken(tokens, progressbar = FALSE)
  vocab = create_vocabulary(it) %>% 
    prune_vocabulary(term_count_min = 10)
  vectorizer = vocab_vectorizer(vocab)
  tcm = create_tcm(it, vectorizer, skip_grams_window = 5L)
  #Sometimes errors "cost too big try using smaller learning rate"
  glove = GlobalVectors$new(word_vectors_size = 300, vocabulary = vocab, x_max = 10, learning_rate = 0.1)
  wv_main = glove$fit_transform(tcm, n_iter = 50, convergence_tol = 0.001)
  wv_context = glove$components
  dim(wv_context)
  word_vectors = wv_main + t(wv_context)
}
#Load word vectors if they exists if not create them
if (!exists("word_vectors")){
  result = tryCatch(
    {
      word_vectors <<- readRDS("word_vectors.rds")
    },
    error=function(cond) {
      word_vectors <<- create_glove()
      saveRDS(word_vectors, file= "word_vectors.rds")
    }
  )
}


#Cluster the LDA topics created in create_lda
#Paper: Then we convert our topic-word matrix $T$ ($t\times w$) to topic-vector matrix $W$ ($t\times v$) 
#Paper: via matrix multiplication $T$($t\times w$)$V$($w\times v$). 
#Paper:  Finally, we use K-medioids clustering to cluster our topics $W$ ($t\times v$) to k topics.
glove_cluster_LDA_topics <- function(tw_dist = tw_dist_store){
  word_vectors_tw_dist <- word_vectors[colnames(tw_dist),, drop = FALSE]
  tw_dist_glove <<- as.matrix(tw_dist) %*% word_vectors_tw_dist
  library(cluster)
  pam(tw_dist_glove, 20)
}
kmed_all <- glove_cluster_LDA_topics (tw_dist_store)

#install #https://rdrr.io/bioc/gespeR/
#source("https://bioconductor.org/biocLite.R")
#biocLite("gespeR")
library("gespeR")

#Here we investigate the clusters, i.e. provide metrics shown in Table 1 in the paper
compressCluster2 <- function (cluster_id_array=1, tw_dist = tw_dist_store, dt_dist_store=dt_dist_store, 
                              kmed=kmed_all, spliter = log_ff$weekend, words)
  {
  #variables where output is stored
  paper_meanRatios1 <<- NULL #Also supports splitting to groups and computing thetas. Not used in the paper
  paper_meanRatios2 <<- NULL #Also supports splitting to groups and computing thetas. Not used in the paper
  paper_n_clusters <<- NULL #Table 1 Topics
  paper_WordRelevances <<- NULL #Self invented measure of cluster stability. Dropped from the paper. 
  paper_tenWordAvgRel <<- NULL # Same as above but for top ten words
  paper_tenWordPrint <<- NULL # Same as above but for printing
  paper_silhAvg <<- NULL #Table 1 Silhouette
  paper_spear <<- NULL #Table 1 Spearman
  paper_pear <<- NULL #Peason correlation not in the paper
  paper_topicPoints <<- NULL #Self invented measure. Not in the paper
  paper_vocab <<- NULL #Size of vocabulary another possible measure of stability. Not in the paper 
  paper_jacc <<- NULL #Table 1 Jaccard
  paper_rbo <<- NULL #Table 1 RBO
  #get all cluster words and make it a unique list
  for (cluster_id in cluster_id_array){
    #temp <- words[kmed$cluster==cluster_id] %>%  word_tokenizer %>% unlist %>% unique

    grp1 <- dt_dist_store[spliter, ]
    grp2 <- dt_dist_store[!spliter, ]
    ratio1 <- colMeans(grp1) / colMeans(grp2) 
    ratio2 <- colMeans(grp2) / colMeans(grp1)
    
    meanRatio1 <-  mean(ratio1 [kmed$cluster==cluster_id])
    meanRatio2 <-  mean(ratio2 [kmed$cluster==cluster_id])
    paper_meanRatios1 <<-   rbind(paper_meanRatios1, meanRatio1)
    paper_meanRatios2 <<-   rbind(paper_meanRatios2, meanRatio2)
    print(paste("Mean ratio1:",meanRatio1, "Mean ratio2:",meanRatio2))
    
    paper_n_clusters <<-   rbind(paper_n_clusters, sum(kmed$cluster==cluster_id))
    
    
    if (sum(kmed$cluster==cluster_id) > 1){
      WordRelevance <- (sort(colSums(tw_dist[kmed$cluster==cluster_id,])/sum(kmed$cluster==cluster_id), decreasing=TRUE))[1:10]
    } else {
      WordRelevance <- (sort(tw_dist[kmed$cluster==cluster_id,]/sum(kmed$cluster==cluster_id), decreasing=TRUE))[1:10]
    }
    paper_WordRelevances <<- rbind(paper_WordRelevances, data.frame(keyName=names(WordRelevance), value=WordRelevance, 
                                                                    keyValue = paste(names(WordRelevance), WordRelevance), 
                                                                    row.names=NULL, cluster_id=cluster_id, stringsAsFactors = FALSE))
    tenWords <- data.frame(unname(as.list(names(WordRelevance))), stringsAsFactors = FALSE)
    colnames(tenWords) <- c(1:10)
    paper_tenWordPrint <<- rbind(paper_tenWordPrint, tenWords)
    
    paper_tenWordAvgRel <<- rbind(paper_tenWordAvgRel,mean (WordRelevance))
    #silhuette here for each
    paper_silhAvg  <<- rbind(paper_silhAvg,kmed$silinfo$clus.avg.widths[cluster_id])
    print(paste("10 word rel avg", mean (WordRelevance)))
    print(paste("Silh avg", kmed$silinfo$clus.avg.widths[cluster_id]))
    #---------------------------------------------------------
    #Loop for pairwise comparisons
    loop_pear <- NULL
    loop_spear <- NULL
    loop_vocab <- NULL
    loop_jacc <- NULL
    loop_rbo <- NULL
    #Post-print comment. 
    #It appears that we compute everything twice. We compute both half of correlation matrix
    #Loop should be improved for efficiency to  j>i should always hold 
    #However, as we compute means of produced pairs it does not matter
    for(i in which(kmed$cluster==cluster_id)){
      #print(names(sort(tw_dist_store[i,], decreasing = TRUE)[1:10]))
      for(j in which(kmed$cluster==cluster_id)){
        #print(names(sort(tw_dist_store[i,], decreasing = TRUE)[1:10]))
        if (i != j){#for efficiency set i<j in future versions. 
          
          t1 <- rank(-tw_dist[i,])
          t2 <- rank(-tw_dist[j,])
          
          t1[t1>10] <- 11
          t2[t2>10] <- 11
          t1_words <- names(t1[t1<11])
          t2_words <- names(t2[t2<11])
          
          
          #top10words <- unique(c(t1_words,t2_words))
          top10words <- union(t1_words,t2_words)
          t1 <- t1[top10words] 
          t2 <- t2[top10words] 
          pear <- cor(t1, t2)
          spear <- cor(t1, t2, method = "spearman")
          vocab_length <- length(top10words)
          jacc <- length(intersect(t1_words,t2_words))/vocab_length
          
          rbo_measure <- rbo(tw_dist[i,], tw_dist[j,], p = 0.9, k=10)

          
          loop_pear<- rbind(loop_pear, pear)
          loop_spear<- rbind(loop_spear, spear)
          loop_vocab<- rbind(loop_vocab, vocab_length)
          loop_jacc<- rbind(loop_jacc, jacc)
          loop_rbo<- rbind(loop_rbo, rbo_measure)
        }
      }
    }
    paper_pear<<- rbind(paper_pear, mean(loop_pear))
    paper_spear<<- rbind(paper_spear, mean (loop_spear))
    paper_vocab<<- rbind(paper_vocab, mean(loop_vocab))
    paper_jacc<<- rbind(paper_jacc, mean(loop_jacc))
    paper_rbo <<-  rbind(paper_rbo, mean(loop_rbo))
    print(paste("pearson avg", mean(loop_pear)))
    print(paste("spearman avg", mean (loop_spear)))
    print(paste("vocab avg", mean (loop_vocab)))
    print(paste("Jacc avg", mean (loop_jacc)))
    print(paste("RBO avg", mean (loop_rbo)))
    #------------------------------------------------------------------------------
    #point wise computation max 55 10-9-8-...
    #points------------------------------------------
    #temp_topics <- which(kmed_all$cluster==cluster_id)
    #temp2 <- tw_dist_store[temp_topics,]
    
    topics <- tw_dist[kmed$cluster==cluster_id,]
    
    #dim(temp3)
    topics <- apply(-topics, 1, rank)
    topics[topics>10] <- 11
    #keep rowming
    
    #install.packages("matrixStats")
    library(matrixStats)
    rows_to_lose <- which(rowMins(topics)>10)
    
    #temp3 <- apply(-temp2, 1, rank)
    topics <- topics[-rows_to_lose,]
    topics <- abs(topics-11)#convert to points 1 -> 10 2->9 11->0
    topics_points <- sort(colSums(t(topics))/sum(kmed$cluster==cluster_id), decreasing=TRUE)[1:10]

    
    paper_topicPoints<<- rbind(paper_topicPoints, sum(topics_points^2)/385)
    print(paste("Points ratio", sum(topics_points^2)/385))
    print(topics_points)
    print(topics_points^2)
    print(sort(WordRelevance, decreasing =TRUE))
  }
}
compressCluster2(1:20, tw_dist_store, dt_dist_store, kmed_all, log_ff$weekend, vocab$term)

#Bind needed measures together
cluster_measures <- cbind(paper_tenWordAvgRel, paper_silhAvg, paper_pear, paper_spear, paper_topicPoints,paper_rbo, paper_vocab,paper_jacc, paper_n_clusters)
cluster_measures <- cbind(paper_silhAvg, paper_spear, paper_jacc, paper_topicPoints,paper_rbo)
#Do measures correlate with each other?
cor(cluster_measures)

#Not used in current paper. Could be used for splitting to two groups
#weekend <- doc_topic_distr[!log_ff$office.hours, ]
#week <- doc_topic_distr[log_ff$office.hours, ]
#weekend <- doc_topic_distr[!log_ff$hired, ]
#week <- doc_topic_distr[log_ff$hired, ]
#March 22, 2011as.Date("2011-03-22")
#weekend <- doc_topic_distr[!log_ff$rapid.release, ]
#week <- doc_topic_distr[log_ff$rapid.release, ]

#Print all needed measures as Latex table
paper_table <- cbind(paper_silhAvg,
                     paper_spear, 
                     paper_jacc, 
                     paper_rbo, 
                     paper_n_clusters, paper_tenWordPrint)
colnames(paper_table) <- c("Silhouette", "Spearman",  "Jaccard", "RBO", "Topics",  1:10)
paper_table <- paper_table[order(-paper_table$RBO),]
paper_table1 <- paper_table[1:10,]
paper_table2 <- paper_table[11:20,]
#paper_table2 <- paper_table2[order(-paper_table2$Ratio2),,]
rownames(paper_table1) <- 1:10
rownames(paper_table2) <- 1:10

colnames(paper_table2) <- c("Silhouette", "Spearman",  "Jaccard", "RBO", "Topics",  1:10)
colnames(paper_table1) <-  c("Silhouette",  "Spearman",  "Jaccard", "RBO", "Topics",  1:10)

paper_table1$Silhouette <-  format(round(paper_table1$Silhouette, 3), nsmall = 3)
paper_table1$Spearman <-  format(round(paper_table1$Spearman, 3), nsmall = 3)
paper_table1$RBO <-  format(round(paper_table1$RBO, 3), nsmall = 3)
paper_table1$Jaccard <-  format(round(paper_table1$Jaccard, 3), nsmall = 3)


paper_table2$Silhouette <-  format(round(paper_table2$Silhouette, 3), nsmall = 3)
paper_table2$Spearman <-  format(round(paper_table2$Spearman, 3), nsmall = 3)
paper_table2$RBO <-  format(round(paper_table2$RBO, 3), nsmall = 3)
paper_table2$Jaccard <-  format(round(paper_table2$Jaccard, 3), nsmall = 3)

colnames(paper_table1) <-  c("Silhouette",  "Spearman",  "Jaccard", "RBO", "Topics",  1:10)
colnames(paper_table2) <- c("Silhouette", "Spearman",  "Jaccard", "RBO", "Topics",  1:10)
#names(paper_table)[names(paper_table) == 'paper_meanRatios'] <- 'Ratio'
#names(paper_table)[names(paper_table) == 'paper_tenWordAvgRel'] <- 'Relevance'
library(xtable)
options(xtable.floating = FALSE)
xtable(t(paper_table1))
xtable(t(paper_table2))

paper_table_small <- rbind (paper_table1[c(1,10),], paper_table2[10,])
xtable(t(paper_table_small))

#Table top 5 from each

#Print the example table
cluster_id <- 10 # 2, 4, 10, Use loop in below to find correct clusters
paper_table_example <<- NULL
for(i in which(kmed_all$cluster==cluster_id)){
  print(names(sort(tw_dist_store[i,], decreasing = TRUE)[1:10]))
  paper_table_example <<- rbind (paper_table_example, names(sort(tw_dist_store[i,], decreasing = TRUE)[1:5]))#Only show top five words. Tables 2-4
}
#We do not have space to show all topics (20) inside a cluster. Select 5. Tables 2-4
xtable(t(paper_table_example[c(1, 4, 8, 12, 16),]))


for (cluster_id in 1:20){
  print(paper_tenWordPrint[cluster_id,])
  #for(i in which(kmed_all$cluster==cluster_id)){
  #  print(names(sort(tw_dist_store[i,], decreasing = TRUE)[1:10]))
  #}
  print("--------------------------------------------------")
}
