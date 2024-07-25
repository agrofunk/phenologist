### the Phenologist
The mission is to combine a bunch of yearly Landsat time-series datasets, calculate a vegetation index, such as kNDVI, perform some filtering, gap-filling and smoothening and then, finally, get some really cool phenology metrics. 

#### Challenges
1. the algorithm, as far as I know, run on a yearly basis, so I have to further restrict the season within a year to capture the phenology
2. that gets even more complicated in irrigated areas because there are more crop rotations, meaning more seasons.