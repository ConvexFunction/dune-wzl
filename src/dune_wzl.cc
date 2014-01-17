// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"     
#endif

#include<stdexcept>
#include<iostream>
#include<memory>
#include<string>
#include<sstream>
#include<vector>

#include<getopt.h>

#include<dune/common/exceptions.hh>
#include<dune/common/fvector.hh>
#include<dune/common/timer.hh>

#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk/vtkwriter.hh>
#include<dune/grid/uggrid.hh>

#include<dune/istl/bvector.hh>
#include<dune/istl/operators.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/io.hh>
#include<dune/istl/umfpack.hh>

#include<dune/pdelab/finiteelementmap/q1fem.hh>
#include<dune/pdelab/constraints/conforming.hh>
#include<dune/pdelab/gridfunctionspace/vectorgridfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include<dune/pdelab/gridfunctionspace/interpolate.hh>
#include<dune/pdelab/gridfunctionspace/vtk.hh>
#include<dune/pdelab/constraints/common/constraints.hh>
#include<dune/pdelab/common/function.hh>
#include<dune/pdelab/backend/istlvectorbackend.hh>
#include<dune/pdelab/backend/istlmatrixbackend.hh>
#include<dune/pdelab/backend/istlsolverbackend.hh>
#include<dune/pdelab/localoperator/linearelasticity.hh>
#include<dune/pdelab/constraints/constraintsparameters.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>

#include "nasreader.hh"
#include <H5Cpp.h>

using namespace Dune;

const int dim = 3;

typedef BlockVector<FieldVector<double,dim> > VectorType;
typedef BCRSMatrix<FieldMatrix<double,dim,dim> > MatrixType;
typedef FieldVector<double, dim> GlobalVector;

typedef UGGrid<dim> GridType;

typedef PDELab::Q1LocalFiniteElementMap<double,double,dim> FEM;
typedef PDELab::VectorGridFunctionSpace<
  GridType::LeafGridView,
  FEM,
  dim,
  PDELab::ISTLVectorBackend<PDELab::ISTLParameters::static_blocking>,
  PDELab::ISTLVectorBackend<>,
  PDELab::ConformingDirichletConstraints,
  PDELab::EntityBlockedOrderingTag,
  PDELab::DefaultLeafOrderingTag
  > GFS;

typedef GFS::ConstraintsContainer<double>::Type C;

typedef PDELab::GridOperator<GFS,GFS,PDELab::LinearElasticity,PDELab::ISTLMatrixBackend,double,double,double,C,C> GO;
typedef GO::Traits::Domain V;
typedef GO::Jacobian M;
typedef M::BaseT ISTL_M;
typedef V::BaseT ISTL_V;

// constraints parameter class for selecting boundary condition type 
template<typename GridView>
struct Dirichlet
  : public PDELab::DirichletConstraintsParameters
{
  const std::vector<bool>& boundaryMap;
  const GridView& gridView;

  Dirichlet(const std::vector<bool>& map, const GridView& gv) : boundaryMap(map), gridView(gv) {}

  template<typename Intersection>
  bool isDirichlet(const Intersection& intersection,
                   const FieldVector<typename Intersection::ctype, Intersection::dimension-1> & coord
                  ) const
  {
    bool isBoundary = true;

    for (size_t i = 0; isBoundary && i < 4; ++i) {
      const ReferenceElement<double, dim>& referenceElement = ReferenceElements<double, dim>::general(intersection.inside()->type());

      unsigned localIdx = referenceElement.subEntity(intersection.indexInInside(), 1, i, dim);
      unsigned globalIdx = gridView.indexSet().subIndex(*intersection.inside(), localIdx, dim);

      isBoundary = isBoundary && boundaryMap[globalIdx];
    }

    return isBoundary;
  }
};

// Helper function for converting numbers to strings
template<typename T>
std::string toString(const T& t) {
  return static_cast<std::ostringstream*>( &(std::ostringstream() << t) )->str();
}

void print_usage() {
  std::cout << "dune-wzl -i in-file -o out-file"                                               << std::endl
            << "         filenames without file extensions"                                    << std::endl
            << "-i filename: read from filename.nas"                                           << std::endl
            << "-o filename: output to filename.h5"                                            << std::endl
            << "-E x: use E=x"                                                                 << std::endl
            << "-n x: use nu=x"                                                                << std::endl;

}

int main(int argc, char** argv) try
{
  // //////////////////////////////// Read command line ////////////////////////////////
  std::string inName, outName;
  double E(0), nu(0);

  int opt;
  while ((opt = getopt(argc, argv, ":i:o:E:n:")) != EOF) {
    switch (opt) {
    case 'i':
      inName = optarg;
      break;
    case 'o':
      outName = optarg;
      break;
    case 'E':
      std::stringstream(optarg) >> E;
      break;
    case 'n':
      std::stringstream(optarg) >> nu;
      break;
    default:
      print_usage();
      throw std::runtime_error("Passed unknown argument.");
    }
  }

  if (inName.empty() or outName.empty()) {
    print_usage();
    throw std::runtime_error("Missing filename for input or output.");
  }

  if (E < 0 or nu < 0) {
    print_usage();
    throw std::runtime_error("Invalid value for E or nu.");
  }

  // //////////////////////////////// Create and start timer ////////////////////////////////
  Timer watch;

  // //////////////////////////////// Read nas file ////////////////////////////////
  std::string nasFilename(inName+std::string(".nas"));
  std::cout << "Reading " << nasFilename << " ..." << std::endl;

  // Create grid by reading *.nas file
  GridType* grid;

  std::vector<unsigned> rigidNodes;
  std::vector<std::pair<unsigned, GlobalVector> > forces;

  NasReader<double, GridType>::read(nasFilename, grid, rigidNodes, forces);

  // Create bitmap for boundary nodes
  std::vector<bool> rigidNodeMap(grid->size(dim), 0);

  for (size_t i = 0; i < rigidNodes.size(); ++i)
    rigidNodeMap[rigidNodes[i]] = true;

  watch.stop();
  std::cout << "Reading and setting up vectors took " << watch.lastElapsed() << " seconds." << std::endl;

  // //////////////////////////////// Assemble problem ////////////////////////////////
  std::cout << "Now setting up the stiffness matrix ..." << std::endl;
  watch.start();

  // Setup grid function space
  FEM fem;
  GFS gfs(grid->leafView(), fem);

  // make constraints map and initialize it via a Dirichlet<GridView> object
  C cg;
  cg.clear();

  Dirichlet<GridType::LeafGridView> bctype(rigidNodeMap, grid->leafView());
  PDELab::constraints(bctype, gfs, cg);

  // convert from nu, E to lame constants
  const double
    lambda = (nu*E)/((1+nu)*(1-2*nu)),
    mu = E/(2*(1+nu));

  // make grid operator
  PDELab::LinearElasticity lop(lambda, mu, 0);
  GO go(gfs, cg, gfs, cg, lop);

  // make coefficient vector and initialize it from a function
  // coefficient vector just zero here since A does not dependent on some parameter
  V x0(gfs);
  x0 = 0.0;

  // represent operator as a matrix
  M m(go);
  m = 0.0;

  go.jacobian(x0, m);

  // get actual matrix
  ISTL_M stiffnessMatrix = m.base();

  // Output how long setting up the matrix took us
  watch.stop();
  std::cout << "Assembling the stiffness matrix took " << watch.lastElapsed() << " seconds." << std::endl;


  // //////////////////////////////// Compute solution ////////////////////////////////
  std::cout << "Now computing the solutions ..." << std::endl;

  ///// Create a solver
  std::cout << "   Calling UMFPack constructor ... " << std::endl;
  watch.start();
  UMFPack<ISTL_M> solver(stiffnessMatrix);
  watch.stop();
  std::cout << "   Calling the UMFPack constructor took " << watch.lastElapsed() << " seconds." << std::endl;
  watch.start();

  ///// H5 file storing the results
  H5::H5File h5OutFile(outName+std::string(".h5"), H5F_ACC_TRUNC);

  // Various variables necessary for writing the solutions via h5
  const double fillvalue = 0;
  H5::DSetCreatPropList plist;
  plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fillvalue);

  // Create dataspace for file
  const hsize_t dune_all_solution_size[3] = {forces.size(), dim, grid->size(dim)};
  H5::DataSpace out_space(3, dune_all_solution_size);
  H5::DataSet out_dataset(h5OutFile.createDataSet("Dune Solution", H5::PredType::NATIVE_DOUBLE, out_space, plist));

  hsize_t dune_solution_start[3] = {0, 0, 0};

  // Create dataspace for data in memory
  const hsize_t dune_single_solution_size[3] = {1, dim, grid->size(dim)};
  H5::DataSpace mem_out_space(3, dune_single_solution_size);
  mem_out_space.selectHyperslab(H5S_SELECT_SET, dune_single_solution_size, dune_solution_start);

  ///// Apply the solver
  ISTL_V x(V(gfs).base());
  ISTL_V rhs(V(gfs).base());

  // Object storing some statistics about the solving process
  InverseOperatorResult statistics;

  for (hsize_t& k = dune_solution_start[0]; k < forces.size(); ++k) {
    x = 0;

    rhs = 0;
    rhs[forces[k].first] = forces[k].second;

    solver.apply(x, rhs, statistics);

    out_space.selectHyperslab(H5S_SELECT_SET, dune_single_solution_size, dune_solution_start);
    out_dataset.write(&(x[0][0]), H5::PredType::NATIVE_DOUBLE, mem_out_space, out_space);
  }

  h5OutFile.close();

  // Done.
  watch.stop();
  std::cout << "Getting the solutions from decomposed matrix and writing the H5 file took " << watch.lastElapsed() << " seconds." << std::endl;
  std::cout << "Total time: " << watch.elapsed() << " seconds." << std::endl;

  return 0;
 }
 catch (Exception &e){
   std::cerr << "Error: " << e << std::endl;
   return 1;
 }
