****** BEFORE INSTALLATION ********
-> For high performance on KNL with MCDRAM mode, Falcon uses the concept of Scratchpad memory to work on,
   which is one time generated and used multiple times throughout the network if there are multiple conv modules
   conneted back to back.
-> As a result the user needs to set the MAX_BATCH, MAX_IMAGE_CHANNELS, MAX_IROWS, MAX_FILTERs and MAX_FILTER_CHANNELS
   parameters in the include/falcon.h file, which allows falcon to allocate max memory that will be used by the user
-> By default these parameters are set w.r.t VGG-16 D network, which can be used as a reference to understand these parameters



*********** USAGE *****************
->falcon_init_lib(); //init falcon library, allocate scratch memory
->fal_conv(M,image,irows,C,filter,K,batch,out);  // conv API
->falcon_free_lib();  // free the scratch pad memory

Refer vgg_winograd.c in the example/ for usage, and to run the example program
> ./run.sh 0 
> ./run.sh 1       ...for run and verify 


********** INSTALLATION ***********
> ./clean.sh
>./install.sh

 
