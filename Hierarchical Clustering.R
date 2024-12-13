################################
#Cluster Analysis
##############################
library(readxl)   
mydata <- read_excel("C:/Users/Elliot/OneDrive/Desktop/States.xlsx")
head(mydata)


#Scatter plots to view some of the variables
plot(mydata)


#Normalization
z <- mydata[,-c(1,1)]
m <- apply(z,2,mean)
s <- apply(z,2,sd)
z <- scale(z,m,s)

#Calculate the euclidean distance
distance <- dist(z)
distance
print(distance, digits = 3)

#Clustering Dendogram
h <- hclust(distance)
plot(h, labels = mydata$City, hang = -1)

# Clustering Dendogram(average)
h1 <- hclust(distance, method = "average")
plot(h1, labels = mydata$City, hang = -1)

#Actual distances
member.1 <- cutree(h1,3)
aggregate(mydata[,-c(1,1)],list(member.1),mean)
