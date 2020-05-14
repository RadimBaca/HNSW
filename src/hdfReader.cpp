#include "hdfReader.h"


//////////////////////////////////////////////////////////////
///////////////////////   HDF5 fun    ////////////////////////
//////////////////////////////////////////////////////////////


void hdfReader::getDimensions(const char* filename, const char* datasetname, hsize_t(*dimensions)[2])
{
    try
    {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(datasetname);
        DataSpace dataspace = dataset.getSpace(); // Get dataspace of the dataset.
        int rank = dataspace.getSimpleExtentNdims(); // Get the number of dimensions in the dataspace.
        int ndims = dataspace.getSimpleExtentDims(*dimensions, NULL);
    }  // end of try block
    catch (FileIException error) { error.printErrorStack();        return; }
    catch (DataSetIException error) { error.printErrorStack();        return; }
    catch (DataSpaceIException error) { error.printErrorStack();        return; }
    catch (DataTypeIException error) { error.printErrorStack();        return; }
}


void hdfReader::readData(const char* filename, const char* datasetname, float* data)
{
    hid_t           file, dset;           /* Handle */
    herr_t          status;
    /*
    * Open file and initialize the operator data structure.
    */
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file, datasetname, H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /*
    * Close and release resources.
    */
    status = H5Dclose(dset);
    status = H5Fclose(file);
}


void hdfReader::readData(const char* filename, const char* datasetname, int* data)
{
    hid_t           file, dset;           /* Handle */
    herr_t          status;
    /*
    * Open file and initialize the operator data structure.
    */
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen(file, datasetname, H5P_DEFAULT);
    status = H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /*
    * Close and release resources.
    */
    status = H5Dclose(dset);
    status = H5Fclose(file);
}