# distutils: sources = src/densecrf_wrapper.cpp

cimport numpy as np

cdef extern from "../include/densecrf_wrapper.h":
    cdef cppclass DenseCRFWrapper:
        DenseCRFWrapper(int, int, int) except +
        void set_unary_energy(float*)
        void add_pairwise_energy(float,
                                 float, float,
                                 float, float, float,
                                 float,
                                 float, float,
                                 unsigned char*)
        void map(int, int*)
        void inference(int, float*)
        int npixels()
        int nlabels()

cdef class DenseCRF:
    cdef DenseCRFWrapper *thisptr

    def __cinit__(self, int W, int H, int nlabels):
        self.thisptr = new DenseCRFWrapper(W, H, nlabels)

    def __dealloc__(self):
        del self.thisptr

    def set_unary_energy(self, float[:] unary_costs):
        self.thisptr.set_unary_energy(&unary_costs[0])

    def add_pairwise_energy(self,
                            float w1,
                            float theta_alpha_1, float theta_alpha_2,
                            float theta_betta_1, float theta_betta_2, float theta_betta_3,
                            float w2,
                            float theta_gamma_1, float theta_gamma_2,
                            unsigned char[:] im):
        self.thisptr.add_pairwise_energy(
            w1,
            theta_alpha_1, theta_alpha_2,
            theta_betta_1, theta_betta_2, theta_betta_3,
            w2,
            theta_gamma_1, theta_gamma_2,
            &im[0]
        )

    def map(self, int n_iters=10):
        import numpy as np
        labels = np.empty(self.thisptr.npixels(), dtype=np.int32)
        cdef int[::1] labels_view = labels
        self.thisptr.map(n_iters, &labels_view[0])
        return labels

    def inference(self, int n_iters=10):
        import numpy as np
        probs = np.empty(self.thisptr.npixels() * self.thisptr.nlabels(), dtype=np.float32)
        cdef float[::1] probs_view = probs
        self.thisptr.inference(n_iters, &probs_view[0])
        return probs
