library(tidyverse)
library(data.table)
library(terra)

#Pixel Per Degree
ppd=128
#Kilometer per Pixel
kmpx=0.2369011752
#Pixel Per Kilometer
pxkm=kmpx**-1
#Kilometer per Degree
kmd=ppd*kmpx
#Degrees per Kilometer
dkm=kmd**-1
#PixelsPerDegree (For image data)
gppd=16
#kilometers per pixel(For image data)
gkmpix=1895/1000
gkmdeg=gppd*gkmpix
gdkm=gkmdeg**(-1)

#Gravity disturbance
moon <- rast("C:/Users/Victor/Desktop/Guachu Tesis/gggrx_1200a_dist_l1200.tif")
plot(moon)
points(0,0, col = "red")

#DEM Well Known Text
WKT <- crs(moon)


#Crater Database
Robbins <- data.table::fread('C:/Users/Victor/Desktop/Guachu Tesis/Lunar Crater Database Robbins 2018.csv',
                          data.table = FALSE,
                          nThread = 12)

str(Robbins)
summary(Robbins)

# Transform LON_CIRC_IMG coordinated from (0 - 360) range to (-180 - 180) range

Robbins <- Robbins %>%
            mutate(LON_CIRC_IMG2=case_when(
                                            LON_CIRC_IMG>180 ~ LON_CIRC_IMG-360,
                                            TRUE             ~ LON_CIRC_IMG
                                          )
  )

#Create GeoDataFrame


Robbins <- Robbins %>%
  mutate(LON_CIRC_IMG_ADJ=LON_CIRC_IMG2-(DIAM_CIRC_IMG/2)*dkm,
         LAT_CIRC_IMG_ADJ=LAT_CIRC_IMG-(DIAM_CIRC_IMG/2)*dkm,
         SHAPE = (DIAM_CIRC_IMG/2)*dkm)



for (i in 1:nrow(Robbins)) {
  
  crater <- crop(moon,buffer(vect(cbind(Robbins$LON_CIRC_IMG_ADJ[i],Robbins$LAT_CIRC_IMG_ADJ[i])),Robbins$SHAPE[i]),mask=TRUE, snap="near")
  writeRaster(crater,paste0("C:/Users/Victor/Desktop/Guachu Tesis/Crater Data/",Robbins$CRATER_ID[i],".tif"),overwrite=TRUE)
  png(paste0("C:/Users/Victor/Desktop/Guachu Tesis/Plot/",Robbins$CRATER_ID[i],".png"))
  plot(crater , legend =FALSE,main=paste0(Robbins$CRATER_ID[i]))
  dev.off()
}

# Read a single file
CraterX <- rast("C:/Users/Victor/Desktop/Guachu Tesis/Crater Data/01-1-000502.tif")
plot(CraterX)
