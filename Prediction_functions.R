# marker_genes is a vector of gene names
# family dict is a named vector of family numbers and cell barcode
# fois is a vector of family numbers
# norm_data is the normalized readcount matrix.

#Change correlation from spearman(non-linear) to pearson(linear)

quantify_clusters = function(norm_data, marker_genes, family_dict, fois, N=2, iterations=20) # maybe filter genes as we go down?
{
  #Get norm data with only subset of genes and one present in current data set
  genes = intersect(marker_genes, rownames(norm_data))
  norm_data = norm_data[genes,]
  norm_data = norm_data[,colSums(norm_data)!=0] #Remove cell without gene expression
  #-> python do that outside the function
  # we could also remove genes that are absent in half of the cells -> we do that on python script 
  
  #Only keep cells that are from a family of interest
  cells = intersect(names(family_dict)[family_dict %in% fois], colnames(norm_data))
  norm_data = norm_data[,cells]
  #-> python do that outside the function
    
  #Compute pearson's correlation
  corr_expr_raw = calculate_correlations(t(norm_data))$r; corr_expr = (1 - corr_expr_raw)/2
  
  #Create empty matrix of size (#cells x iteration) with name of cells (barcodes) as row names
  cell_clusters = data.matrix(matrix(0, nrow=ncol(norm_data), ncol=iterations)); rownames(cell_clusters) = colnames(norm_data)
    
  #Create empty matrix of size (#cells x iteration) with name of cells (barcode) as row names
  cell_clusters_correlation = data.matrix(matrix(NA, nrow=ncol(norm_data), ncol=iterations)); rownames(cell_clusters_correlation) = colnames(norm_data)

  #Put 1 in all first iteration column
  cell_clusters[,1] = rep(1, ncol(norm_data))
  #Put the mean correlation in all first iteration column
  cell_clusters_correlation[,1] = mean(corr_expr[lower.tri(corr_expr, diag=F)])
    

  for (i in seq(2:iterations))
  {
    for (cluster in setdiff(unique(cell_clusters[,(i-1)]),0))
    {
      cells_in_cluster = rownames(cell_clusters)[cell_clusters[,(i-1)]==cluster]
      if (length(cells_in_cluster) >= 3)
      {
        correlation = mean((corr_expr_raw[cells_in_cluster,cells_in_cluster])[lower.tri(corr_expr_raw[cells_in_cluster,cells_in_cluster], diag=F)])
        cell_clusters_correlation[cells_in_cluster,(i-1)] = rep(correlation, length(cells_in_cluster))
        corr_expr_subset = corr_expr[cells_in_cluster,cells_in_cluster]
        clustering = cutree(hclust(as.dist(corr_expr_subset), method = "ward.D2", ), k=N, )
        cell_clusters[names(clustering),i] = as.vector(clustering) + max(c(0, cell_clusters[,i]), na.rm=T)
      }
      else 
      {
        cell_clusters[cells_in_cluster,i] = 0
      }
    }
  }
  co_clustering = matrix(0, nrow=ncol(norm_data), ncol=ncol(norm_data)); rownames(co_clustering) = colnames(norm_data); colnames(co_clustering) = colnames(norm_data)
  cell_clusters[cell_clusters==0] = NA
  for (i in seq(1, iterations))
  {
    to_add = (1.0*outer(cell_clusters[,i], cell_clusters[,i], FUN='==')); to_add[is.na(to_add)] = 0
    co_clustering = co_clustering + to_add
  }
  return(list(cell_clusters, co_clustering, cell_clusters_correlation))
}

test_prediction_multiple_overlap_3 <- function(result1, result2, result3, family_dict)
{
  cell_clusters_correlation1 = result1[[3]]
  co_clustering1 = result1[[2]]
  cell_clusters1 = result1[[1]]
  real_family_matrix1 = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
  cell_clusters_unique_name1 = cell_clusters1; for (colname in 1:20){cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname])}
  clustersize_dict1 = table(cell_clusters_unique_name1)
  
  smallest_clusters1 = names(clustersize_dict1)[clustersize_dict1 %in% c(2,3)]
  best_prediction1 = real_family_matrix1; best_prediction1[T] <- F
  for (cluster1 in smallest_clusters1){cells_in_cluster1 = rownames(best_prediction1)[rowSums(cell_clusters_unique_name1==cluster1, na.rm=T)>0]
  best_prediction1[cells_in_cluster1,cells_in_cluster1] <- T}
  diag(best_prediction1) = F; diag(real_family_matrix1) = F
  
  cell_clusters_correlation2 = result2[[3]]
  co_clustering2 = result2[[2]]
  cell_clusters2 = result2[[1]]
  real_family_matrix2 = outer(family_dict[rownames(result2[[2]])], family_dict[rownames(result2[[2]])], FUN='==')
  cell_clusters_unique_name2 = cell_clusters2; for (colname in 1:20){cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname])}
  clustersize_dict2 = table(cell_clusters_unique_name2)
  
  smallest_clusters2 = names(clustersize_dict2)[clustersize_dict2 %in% c(2,3)]
  best_prediction2 = real_family_matrix2; best_prediction2[T] <- F
  for (cluster2 in smallest_clusters2){cells_in_cluster2 = rownames(best_prediction2)[rowSums(cell_clusters_unique_name2==cluster2, na.rm=T)>0]
  best_prediction2[cells_in_cluster2,cells_in_cluster2] <- T}
  diag(best_prediction2) = F; diag(real_family_matrix2) = F
  
  cell_clusters_correlation3 = result3[[3]]
  co_clustering3 = result3[[2]]
  cell_clusters3 = result3[[1]]
  real_family_matrix3 = outer(family_dict[rownames(result3[[2]])], family_dict[rownames(result3[[2]])], FUN='==')
  cell_clusters_unique_name3 = cell_clusters3; for (colname in 1:20){cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname])}
  clustersize_dict3 = table(cell_clusters_unique_name3)
  
  smallest_clusters3 = names(clustersize_dict3)[clustersize_dict3 %in% c(2,3)]
  best_prediction3 = real_family_matrix3; best_prediction3[T] <- F
  for (cluster3 in smallest_clusters3){cells_in_cluster3 = rownames(best_prediction3)[rowSums(cell_clusters_unique_name3==cluster3, na.rm=T)>0]
  best_prediction3[cells_in_cluster3,cells_in_cluster3] <- T}
  diag(best_prediction3) = F; diag(real_family_matrix3) = F
  
  
  best_prediction = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
  best_prediction[T] <- F
  for (cell in 1:length(rownames(best_prediction))){
    for (other_cell in 1:length(rownames(best_prediction))){
      if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T)) {best_prediction[cell,other_cell] <- T}
    }}
  #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction2[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction1[cell,other_cell]==T)) {best_prediction[cell,other_cell] <- T}
  #}}
  diag(best_prediction) = F
  
  return(c(sum(best_prediction & real_family_matrix1), sum(best_prediction & (!real_family_matrix1)), sum(real_family_matrix1)))
}












 #


test_prediction_multiple_overlap_5 <- function(result1, result2, result3, result4, result5, outof, clustersize, family_dict)
{
  cell_clusters_correlation1 = result1[[3]]
  co_clustering1 = result1[[2]]
  cell_clusters1 = result1[[1]]
  real_family_matrix1 = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
  cell_clusters_unique_name1 = cell_clusters1; for (colname in 1:20){cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname])}
  clustersize_dict1 = table(cell_clusters_unique_name1)
  
  smallest_clusters1 = names(clustersize_dict1)[clustersize_dict1 %in% clustersize]
  best_prediction1 = real_family_matrix1; best_prediction1[T] <- F
  for (cluster1 in smallest_clusters1){cells_in_cluster1 = rownames(best_prediction1)[rowSums(cell_clusters_unique_name1==cluster1, na.rm=T)>0]
  best_prediction1[cells_in_cluster1,cells_in_cluster1] <- T}
  diag(best_prediction1) = F; diag(real_family_matrix1) = F
  best_prediction1[best_prediction1==T] <- 1
  
  cell_clusters_correlation2 = result2[[3]]
  co_clustering2 = result2[[2]]
  cell_clusters2 = result2[[1]]
  real_family_matrix2 = outer(family_dict[rownames(result2[[2]])], family_dict[rownames(result2[[2]])], FUN='==')
  cell_clusters_unique_name2 = cell_clusters2; for (colname in 1:20){cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname])}
  clustersize_dict2 = table(cell_clusters_unique_name2)
  
  smallest_clusters2 = names(clustersize_dict2)[clustersize_dict2 %in% clustersize]
  best_prediction2 = real_family_matrix2; best_prediction2[T] <- F
  for (cluster2 in smallest_clusters2){cells_in_cluster2 = rownames(best_prediction2)[rowSums(cell_clusters_unique_name2==cluster2, na.rm=T)>0]
  best_prediction2[cells_in_cluster2,cells_in_cluster2] <- T}
  diag(best_prediction2) = F; diag(real_family_matrix2) = F
  best_prediction2[best_prediction2==T] <- 1
  
  cell_clusters_correlation3 = result3[[3]]
  co_clustering3 = result3[[2]]
  cell_clusters3 = result3[[1]]
  real_family_matrix3 = outer(family_dict[rownames(result3[[2]])], family_dict[rownames(result3[[2]])], FUN='==')
  cell_clusters_unique_name3 = cell_clusters3; for (colname in 1:20){cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname])}
  clustersize_dict3 = table(cell_clusters_unique_name3)
  
  smallest_clusters3 = names(clustersize_dict3)[clustersize_dict3 %in% clustersize]
  best_prediction3 = real_family_matrix3; best_prediction3[T] <- F
  for (cluster3 in smallest_clusters3){cells_in_cluster3 = rownames(best_prediction3)[rowSums(cell_clusters_unique_name3==cluster3, na.rm=T)>0]
  best_prediction3[cells_in_cluster3,cells_in_cluster3] <- T}
  diag(best_prediction3) = F; diag(real_family_matrix3) = F
  best_prediction3[best_prediction3==T] <- 1
  
  cell_clusters_correlation4 = result4[[3]]
  co_clustering4 = result4[[2]]
  cell_clusters4 = result4[[1]]
  real_family_matrix4 = outer(family_dict[rownames(result4[[2]])], family_dict[rownames(result4[[2]])], FUN='==')
  cell_clusters_unique_name4 = cell_clusters4; for (colname in 1:20){cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname])}
  clustersize_dict4 = table(cell_clusters_unique_name4)
  
  smallest_clusters4 = names(clustersize_dict4)[clustersize_dict4 %in% clustersize]
  best_prediction4 = real_family_matrix4; best_prediction4[T] <- F
  for (cluster4 in smallest_clusters4){cells_in_cluster4 = rownames(best_prediction4)[rowSums(cell_clusters_unique_name4==cluster4, na.rm=T)>0]
  best_prediction4[cells_in_cluster4,cells_in_cluster4] <- T}
  diag(best_prediction4) = F; diag(real_family_matrix4) = F
  best_prediction4[best_prediction4==T] <- 1
  
  cell_clusters_correlation5 = result5[[3]]
  co_clustering5 = result5[[2]]
  cell_clusters5 = result5[[1]]
  real_family_matrix5 = outer(family_dict[rownames(result5[[2]])], family_dict[rownames(result5[[2]])], FUN='==')
  cell_clusters_unique_name5 = cell_clusters5; for (colname in 1:20){cell_clusters_unique_name5[!is.na(cell_clusters_unique_name5[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name5[!is.na(cell_clusters_unique_name5[,colname]),colname])}
  clustersize_dict5 = table(cell_clusters_unique_name5)
  
  smallest_clusters5 = names(clustersize_dict5)[clustersize_dict5 %in% clustersize]
  best_prediction5 = real_family_matrix5; best_prediction5[T] <- F
  for (cluster5 in smallest_clusters5){cells_in_cluster5 = rownames(best_prediction5)[rowSums(cell_clusters_unique_name5==cluster5, na.rm=T)>0]
  best_prediction5[cells_in_cluster5,cells_in_cluster5] <- T}
  diag(best_prediction5) = F; diag(real_family_matrix5) = F
  best_prediction5[best_prediction5==T] <- 1
  
  #cell_clusters_correlation6 = result6[[3]]
  #co_clustering6 = result6[[2]]
  #cell_clusters6 = result6[[1]]
  #real_family_matrix6 = outer(family_dict[rownames(result6[[2]])], family_dict[rownames(result6[[2]])], FUN='==')
  #cell_clusters_unique_name6 = cell_clusters6; for (colname in 1:20){cell_clusters_unique_name6[!is.na(cell_clusters_unique_name6[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name6[!is.na(cell_clusters_unique_name6[,colname]),colname])}
  #clustersize_dict6 = table(cell_clusters_unique_name6)
  
  #smallest_clusters6 = names(clustersize_dict6)[clustersize_dict6 %in% c(2,3,4,5)]
  #best_prediction6 = real_family_matrix6; best_prediction6[T] <- F
  #for (cluster6 in smallest_clusters6){cells_in_cluster6 = rownames(best_prediction6)[rowSums(cell_clusters_unique_name6==cluster6, na.rm=T)>0]
  #best_prediction6[cells_in_cluster6,cells_in_cluster6] <- T}
  #diag(best_prediction6) = F; diag(real_family_matrix6) = F
  #best_prediction6[best_prediction6==T] <- 1
  
  best_prediction = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
  best_prediction[T] <- F
  for (cell in 1:length(rownames(best_prediction))){
    for (other_cell in 1:length(rownames(best_prediction))){
      #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction4[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)) {best_prediction[cell,other_cell] <- T}
      #}}
      #if (best_prediction1[cell,other_cell]==1) {best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell]+ best_prediction5[cell,other_cell])
      best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell]+ best_prediction5[cell,other_cell])
      
    }}
  #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction4[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T)|
  #(best_prediction2[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T | (best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)| (best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction5[cell,other_cell]==T))) {best_prediction[cell,other_cell] <- T}
  #}}
  diag(best_prediction) = F
  
  #return(best_prediction)
  
  return(c(sum(best_prediction %in% outof & real_family_matrix1==T), sum(best_prediction %in% outof & real_family_matrix1!=T), sum(real_family_matrix1)))
}


test_prediction_multiple_overlap_4 <- function(result1, result2, result3, result4, outof, clustersize, family_dict)
{
    cell_clusters_correlation1 = result1[[3]]
    co_clustering1 = result1[[2]]
    cell_clusters1 = result1[[1]]
    real_family_matrix1 = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
    cell_clusters_unique_name1 = cell_clusters1; for (colname in 1:20){cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname])}
    clustersize_dict1 = table(cell_clusters_unique_name1)
    
    smallest_clusters1 = names(clustersize_dict1)[clustersize_dict1 %in% clustersize]
    best_prediction1 = real_family_matrix1; best_prediction1[T] <- F
    for (cluster1 in smallest_clusters1){cells_in_cluster1 = rownames(best_prediction1)[rowSums(cell_clusters_unique_name1==cluster1, na.rm=T)>0]
        best_prediction1[cells_in_cluster1,cells_in_cluster1] <- T}
    diag(best_prediction1) = F; diag(real_family_matrix1) = F
    best_prediction1[best_prediction1==T] <- 1
    
    cell_clusters_correlation2 = result2[[3]]
    co_clustering2 = result2[[2]]
    cell_clusters2 = result2[[1]]
    real_family_matrix2 = outer(family_dict[rownames(result2[[2]])], family_dict[rownames(result2[[2]])], FUN='==')
    cell_clusters_unique_name2 = cell_clusters2; for (colname in 1:20){cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname])}
    clustersize_dict2 = table(cell_clusters_unique_name2)
    
    smallest_clusters2 = names(clustersize_dict2)[clustersize_dict2 %in% clustersize]
    best_prediction2 = real_family_matrix2; best_prediction2[T] <- F
    for (cluster2 in smallest_clusters2){cells_in_cluster2 = rownames(best_prediction2)[rowSums(cell_clusters_unique_name2==cluster2, na.rm=T)>0]
        best_prediction2[cells_in_cluster2,cells_in_cluster2] <- T}
    diag(best_prediction2) = F; diag(real_family_matrix2) = F
    best_prediction2[best_prediction2==T] <- 1
    
    cell_clusters_correlation3 = result3[[3]]
    co_clustering3 = result3[[2]]
    cell_clusters3 = result3[[1]]
    real_family_matrix3 = outer(family_dict[rownames(result3[[2]])], family_dict[rownames(result3[[2]])], FUN='==')
    cell_clusters_unique_name3 = cell_clusters3; for (colname in 1:20){cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname])}
    clustersize_dict3 = table(cell_clusters_unique_name3)
    
    smallest_clusters3 = names(clustersize_dict3)[clustersize_dict3 %in% clustersize]
    best_prediction3 = real_family_matrix3; best_prediction3[T] <- F
    for (cluster3 in smallest_clusters3){cells_in_cluster3 = rownames(best_prediction3)[rowSums(cell_clusters_unique_name3==cluster3, na.rm=T)>0]
        best_prediction3[cells_in_cluster3,cells_in_cluster3] <- T}
    diag(best_prediction3) = F; diag(real_family_matrix3) = F
    best_prediction3[best_prediction3==T] <- 1
    
    cell_clusters_correlation4 = result4[[3]]
    co_clustering4 = result4[[2]]
    cell_clusters4 = result4[[1]]
    real_family_matrix4 = outer(family_dict[rownames(result4[[2]])], family_dict[rownames(result4[[2]])], FUN='==')
    cell_clusters_unique_name4 = cell_clusters4; for (colname in 1:20){cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname])}
    clustersize_dict4 = table(cell_clusters_unique_name4)
    
    smallest_clusters4 = names(clustersize_dict4)[clustersize_dict4 %in% clustersize]
    best_prediction4 = real_family_matrix4; best_prediction4[T] <- F
    for (cluster4 in smallest_clusters4){cells_in_cluster4 = rownames(best_prediction4)[rowSums(cell_clusters_unique_name4==cluster4, na.rm=T)>0]
        best_prediction4[cells_in_cluster4,cells_in_cluster4] <- T}
    diag(best_prediction4) = F; diag(real_family_matrix4) = F
    best_prediction4[best_prediction4==T] <- 1
    
    
    best_prediction = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
    best_prediction[T] <- F
    for (cell in 1:length(rownames(best_prediction))){
        for (other_cell in 1:length(rownames(best_prediction))){
            #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction4[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)) {best_prediction[cell,other_cell] <- T}
            #}}
            #if (best_prediction1[cell,other_cell]==1) {best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell]+ best_prediction5[cell,other_cell])
            best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell])
            
        }}
    #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction4[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T)|
    #(best_prediction2[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T | (best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)| (best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction5[cell,other_cell]==T))) {best_prediction[cell,other_cell] <- T}
    #}}
    diag(best_prediction) = F
    
    #return(best_prediction)
    
    return(c(sum(best_prediction %in% outof & real_family_matrix1==T), sum(best_prediction %in% outof & real_family_matrix1!=T), sum(real_family_matrix1)))
}

test_prediction_multiple_overlap_5_wCellcycle <- function(result1, result2, result3, result4, result5, outof, clustersize, family_dict, phases)
{
    cell_clusters_correlation1 = result1[[3]]
    co_clustering1 = result1[[2]]
    cell_clusters1 = result1[[1]]
    real_family_matrix1 = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
    cell_clusters_unique_name1 = cell_clusters1; for (colname in 1:20){cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name1[!is.na(cell_clusters_unique_name1[,colname]),colname])}
    clustersize_dict1 = table(cell_clusters_unique_name1)
    
    smallest_clusters1 = names(clustersize_dict1)[clustersize_dict1 %in% clustersize]
    best_prediction1 = real_family_matrix1; best_prediction1[T] <- F
    for (cluster1 in smallest_clusters1){cells_in_cluster1 = rownames(best_prediction1)[rowSums(cell_clusters_unique_name1==cluster1, na.rm=T)>0]
        best_prediction1[cells_in_cluster1,cells_in_cluster1] <- T}
    diag(best_prediction1) = F; diag(real_family_matrix1) = F
    best_prediction1[best_prediction1==T] <- 1
    
    cell_clusters_correlation2 = result2[[3]]
    co_clustering2 = result2[[2]]
    cell_clusters2 = result2[[1]]
    real_family_matrix2 = outer(family_dict[rownames(result2[[2]])], family_dict[rownames(result2[[2]])], FUN='==')
    cell_clusters_unique_name2 = cell_clusters2; for (colname in 1:20){cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name2[!is.na(cell_clusters_unique_name2[,colname]),colname])}
    clustersize_dict2 = table(cell_clusters_unique_name2)
    
    smallest_clusters2 = names(clustersize_dict2)[clustersize_dict2 %in% clustersize]
    best_prediction2 = real_family_matrix2; best_prediction2[T] <- F
    for (cluster2 in smallest_clusters2){cells_in_cluster2 = rownames(best_prediction2)[rowSums(cell_clusters_unique_name2==cluster2, na.rm=T)>0]
        best_prediction2[cells_in_cluster2,cells_in_cluster2] <- T}
    diag(best_prediction2) = F; diag(real_family_matrix2) = F
    best_prediction2[best_prediction2==T] <- 1
    
    cell_clusters_correlation3 = result3[[3]]
    co_clustering3 = result3[[2]]
    cell_clusters3 = result3[[1]]
    real_family_matrix3 = outer(family_dict[rownames(result3[[2]])], family_dict[rownames(result3[[2]])], FUN='==')
    cell_clusters_unique_name3 = cell_clusters3; for (colname in 1:20){cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name3[!is.na(cell_clusters_unique_name3[,colname]),colname])}
    clustersize_dict3 = table(cell_clusters_unique_name3)
    
    smallest_clusters3 = names(clustersize_dict3)[clustersize_dict3 %in% clustersize]
    best_prediction3 = real_family_matrix3; best_prediction3[T] <- F
    for (cluster3 in smallest_clusters3){cells_in_cluster3 = rownames(best_prediction3)[rowSums(cell_clusters_unique_name3==cluster3, na.rm=T)>0]
        best_prediction3[cells_in_cluster3,cells_in_cluster3] <- T}
    diag(best_prediction3) = F; diag(real_family_matrix3) = F
    best_prediction3[best_prediction3==T] <- 1
    
    cell_clusters_correlation4 = result4[[3]]
    co_clustering4 = result4[[2]]
    cell_clusters4 = result4[[1]]
    real_family_matrix4 = outer(family_dict[rownames(result4[[2]])], family_dict[rownames(result4[[2]])], FUN='==')
    cell_clusters_unique_name4 = cell_clusters4; for (colname in 1:20){cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name4[!is.na(cell_clusters_unique_name4[,colname]),colname])}
    clustersize_dict4 = table(cell_clusters_unique_name4)
    
    smallest_clusters4 = names(clustersize_dict4)[clustersize_dict4 %in% clustersize]
    best_prediction4 = real_family_matrix4; best_prediction4[T] <- F
    for (cluster4 in smallest_clusters4){cells_in_cluster4 = rownames(best_prediction4)[rowSums(cell_clusters_unique_name4==cluster4, na.rm=T)>0]
        best_prediction4[cells_in_cluster4,cells_in_cluster4] <- T}
    diag(best_prediction4) = F; diag(real_family_matrix4) = F
    best_prediction4[best_prediction4==T] <- 1
    
    cell_clusters_correlation5 = result5[[3]]
    co_clustering5 = result5[[2]]
    cell_clusters5 = result5[[1]]
    real_family_matrix5 = outer(family_dict[rownames(result5[[2]])], family_dict[rownames(result5[[2]])], FUN='==')
    cell_clusters_unique_name5 = cell_clusters5; for (colname in 1:20){cell_clusters_unique_name5[!is.na(cell_clusters_unique_name5[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name5[!is.na(cell_clusters_unique_name5[,colname]),colname])}
    clustersize_dict5 = table(cell_clusters_unique_name5)
    
    smallest_clusters5 = names(clustersize_dict5)[clustersize_dict5 %in% clustersize]
    best_prediction5 = real_family_matrix5; best_prediction5[T] <- F
    for (cluster5 in smallest_clusters5){cells_in_cluster5 = rownames(best_prediction5)[rowSums(cell_clusters_unique_name5==cluster5, na.rm=T)>0]
        best_prediction5[cells_in_cluster5,cells_in_cluster5] <- T}
    diag(best_prediction5) = F; diag(real_family_matrix5) = F
    best_prediction5[best_prediction5==T] <- 1
    
    #cell_clusters_correlation6 = result6[[3]]
    #co_clustering6 = result6[[2]]
    #cell_clusters6 = result6[[1]]
    #real_family_matrix6 = outer(family_dict[rownames(result6[[2]])], family_dict[rownames(result6[[2]])], FUN='==')
    #cell_clusters_unique_name6 = cell_clusters6; for (colname in 1:20){cell_clusters_unique_name6[!is.na(cell_clusters_unique_name6[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name6[!is.na(cell_clusters_unique_name6[,colname]),colname])}
    #clustersize_dict6 = table(cell_clusters_unique_name6)
    
    #smallest_clusters6 = names(clustersize_dict6)[clustersize_dict6 %in% c(2,3,4,5)]
    #best_prediction6 = real_family_matrix6; best_prediction6[T] <- F
    #for (cluster6 in smallest_clusters6){cells_in_cluster6 = rownames(best_prediction6)[rowSums(cell_clusters_unique_name6==cluster6, na.rm=T)>0]
    #best_prediction6[cells_in_cluster6,cells_in_cluster6] <- T}
    #diag(best_prediction6) = F; diag(real_family_matrix6) = F
    #best_prediction6[best_prediction6==T] <- 1
    
    best_prediction = outer(family_dict[rownames(result1[[2]])], family_dict[rownames(result1[[2]])], FUN='==')
    best_prediction[T] <- F
    for (cell in 1:length(rownames(best_prediction))){
        for (other_cell in 1:length(rownames(best_prediction))){
            #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction4[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)) {best_prediction[cell,other_cell] <- T}
            #}}
            #if (best_prediction1[cell,other_cell]==1) {best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell]+ best_prediction5[cell,other_cell])
            best_prediction[cell,other_cell]=(best_prediction1[cell,other_cell]+ best_prediction2[cell,other_cell]+ best_prediction3[cell,other_cell]+ best_prediction4[cell,other_cell]+ best_prediction5[cell,other_cell])
            
        }}
    #if ((best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction4[cell,other_cell]==T) | (best_prediction3[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T)|
    #(best_prediction2[cell,other_cell]==T & best_prediction1[cell,other_cell]==T & best_prediction4[cell,other_cell]==T | (best_prediction1[cell,other_cell]==T & best_prediction2[cell,other_cell]==T & best_prediction5[cell,other_cell]==T)| (best_prediction2[cell,other_cell]==T & best_prediction3[cell,other_cell]==T & best_prediction5[cell,other_cell]==T))) {best_prediction[cell,other_cell] <- T}
    #}}
    diag(best_prediction) = F
    
    # Could add here selection on the cell cycle phase
    # By introducing a table as the best_prediction matrix based on same cell cycle phase
    cellcycle_family_matrix = outer(family_dict[rownames(results[[2]])], family_dict[rownames(results[[2]])], FUN='==')
    cellcycle_family_matrix[T] <- F
    for (cell in 1:length(rownames(cellcycle_family_matrix))){
        for (other_cell in 1:length(rownames(cellcycle_family_matrix))){
            if (phases[cell,1]==phases[other_cell,1]){cellcycle_family_matrix[cell,other_cell] <- T}
        }}
    # Now filter the best prediction based on cellcycle_family_matrix
    for (cell in 1:length(rownames(cellcycle_family_matrix))){
        for (other_cell in 1:length(rownames(cellcycle_family_matrix))){
            if (cellcycle_family_matrix[cell,other_cell]== F){best_prediction[cell,other_cell] <- F}
        }}
    
    
    #return(best_prediction)
    
    return(c(sum(best_prediction %in% outof & real_family_matrix1==T), sum(best_prediction %in% outof & real_family_matrix1!=T), sum(real_family_matrix1)))
}

test_prediction <- function(result, family_dict)
{
  cell_clusters_correlation = result[[3]]
  co_clustering = result[[2]]
  cell_clusters = result[[1]]
  real_family_matrix = outer(family_dict[rownames(result[[2]])], family_dict[rownames(result[[2]])], FUN='==')
  cell_clusters_unique_name = cell_clusters; for (colname in 1:20){cell_clusters_unique_name[!is.na(cell_clusters_unique_name[,colname]),colname] = paste0(colname,'_',cell_clusters_unique_name[!is.na(cell_clusters_unique_name[,colname]),colname])}
  clustersize_dict = table(cell_clusters_unique_name)
  
  smallest_clusters = names(clustersize_dict)[clustersize_dict %in% c(2,3)]
  best_prediction = real_family_matrix; best_prediction[T] <- F
  for (cluster in smallest_clusters){cells_in_cluster = rownames(best_prediction)[rowSums(cell_clusters_unique_name==cluster, na.rm=T)>0]
  best_prediction[cells_in_cluster,cells_in_cluster] <- T}
  diag(best_prediction) = F; diag(real_family_matrix) = F
  return(c(sum(best_prediction & real_family_matrix), sum(best_prediction & (!real_family_matrix)), sum(real_family_matrix)))
}
