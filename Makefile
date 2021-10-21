KOKKOS_PATH = extern/kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
EXE_NAME = "main"
EXTRA_INC = -Iextern/plog/include

#SRC = $(wildcard *.cpp)
SRC := full_re.cpp
#SRC := 1darr_lt.cpp
#SRC := alloc.cpp
SRC := $(SRC) timer.cpp

default: build
	echo "Start Build"

KOKKOS_CXX_STANDARD = c++17

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
EXE = ${EXE_NAME}.cuda
# KOKKOS_ARCH = "AMPERE80"
KOKKOS_ARCH = "VOLTA70"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
CXXFLAGS = -DGPUENABLED --expt-relaxed-constexpr
else
CXX = icpc
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "SKX"
ifneq (,$(findstring OpenMP,$(KOKKOS_DEVICES)))
CXXFLAGS = -DOMPENABLED
endif
CXXFLAGS := $(CXXFLAGS) -Ofast -ipo
endif

CXXFLAGS := ${CXXFLAGS} -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
