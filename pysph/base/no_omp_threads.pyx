cpdef int get_number_of_threads():
    return 1

cpdef set_number_of_threads(int n):
    print("OpenMP not available, cannot set number of threads.")

