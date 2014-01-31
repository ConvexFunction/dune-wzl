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

#include<dune/pdelab/finiteelementmap/qkfem.hh>
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

// we have a 3-dimensional problem
const int dim = 3;

// general typedefs
typedef BlockVector<FieldVector<double,dim> > VectorType;
typedef BCRSMatrix<FieldMatrix<double,dim,dim> > MatrixType;
typedef FieldVector<double, dim> GlobalVector;

typedef UGGrid<dim> GridType;
typedef GridType::LeafGridView GV;

// constraints parameter class for selecting boundary condition type 
class Model
  : public Dune::PDELab::LinearElasticityParameterInterface<
  Dune::PDELab::LinearElasticityParameterTraits<GV, double>,
  Model >
{
public:
  typedef Dune::PDELab::LinearElasticityParameterTraits<GV, double> Traits;

  Model(const GV& gv,
        const std::vector<bool>& rigidNodesMap,
        Traits::RangeFieldType l,
        Traits::RangeFieldType m) :
    gridView(gv), boundaryMap(rigidNodesMap), G_(0), lambda_(l), mu_(m)
  {}

  template<typename Intersection>
  bool isDirichlet(const Intersection& intersection,
                   const typename Traits::IntersectionDomainType & coord
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

  void
  u (const Traits::ElementType& e, const Traits::DomainType& x,
     Traits::RangeType & y) const
  {
    y = 0.0;
  }

  Traits::RangeFieldType
  lambda (const Traits::ElementType& e, const Traits::DomainType& x) const
  {
    return lambda_;
  }

  Traits::RangeFieldType
  mu (const Traits::ElementType& e, const Traits::DomainType& x) const
  {
    return mu_;
  }

private:
  const GV& gridView;
  const std::vector<bool>& boundaryMap;

  Traits::RangeType G_;
  Traits::RangeFieldType lambda_;
  Traits::RangeFieldType mu_;
};


// PDELab types we need
typedef PDELab::QkLocalFiniteElementMap<GV,double,double,1> FEM;
typedef PDELab::VectorGridFunctionSpace<
  GV,
  FEM,
  dim,
  PDELab::ISTLVectorBackend<PDELab::ISTLParameters::static_blocking>,
  PDELab::ISTLVectorBackend<>,
  PDELab::ConformingDirichletConstraints,
  PDELab::EntityBlockedOrderingTag,
  PDELab::DefaultLeafOrderingTag
  > GFS;

typedef GFS::ConstraintsContainer<double>::Type C;

typedef PDELab::GridOperator<
  GFS,
  GFS,
  PDELab::LinearElasticity<Model>,
  PDELab::ISTLMatrixBackend,
  double,double,double, C,C> GO;

typedef GO::Traits::Domain V;
typedef GO::Jacobian M;
typedef M::BaseT ISTL_M;
typedef V::BaseT ISTL_V;


// Helper function for converting numbers to strings
template<typename T>
std::string toString(const T& t) {
  return static_cast<std::ostringstream*>( &(std::ostringstream() << t) )->str();
}

void print_usage() {
  std::cout << "dune_wzl_compare -i infile -o outfile"                                         << std::endl
            << "         filenames without file extensions"                                    << std::endl
            << "-i infile : read from infile.nas and infile.h5"                                << std::endl
            << "-o outfile: output to outfile.h5"                                              << std::endl;
}

int main(int argc, char** argv) try
{
  // //////////////////////////////// Read command line ////////////////////////////////
  std::string inName, outName;

  int opt;
  while ((opt = getopt(argc, argv, ":i:o:")) != EOF) {
    switch (opt) {
    case 'i':
      inName = optarg;
      break;
    case 'o':
      outName = optarg;
      break;
    default:
      print_usage();
      throw std::runtime_error("Passed unknown argument.");
    }
  }

  if (inName.empty() or outName.empty())
    throw std::runtime_error("Missing filename for input or output.");


  // //////////////////////////////// Create and start timer ////////////////////////////////
  Timer watch;


  // //////////////////////////////// Read nas file ////////////////////////////////
  std::string nasFilename(inName+std::string(".nas"));
  std::cout << "Reading " << nasFilename << " ..." << std::endl;

  std::vector<unsigned> rigidNodes;
  std::vector<std::pair<unsigned, GlobalVector> > forces;

  // Create grid by reading *.nas file
  GridType* grid;

  NasReader<double, GridType>::read(nasFilename, grid, rigidNodes, forces);

  // Convert nodes to bitmaps
  std::vector<bool> rigidNodeMap(grid->size(dim), 0);

  for (size_t i = 0; i < rigidNodes.size(); ++i)
    rigidNodeMap[rigidNodes[i]] = true;

  std::vector<bool> forceMap(grid->size(dim), 0);

  for (size_t i = 0; i < forces.size(); ++i)
    forceMap[forces[i].first] = true;

  watch.stop();
  std::cout << "Reading and setting up vectors took " << watch.lastElapsed() << " seconds." << std::endl;

  // //////////////////////////////// Prepare vtkWriter ////////////////////////////////
  VTKWriter<GV> vtkWriter(grid->leafGridView());

  vtkWriter.addVertexData(rigidNodeMap, "rigid");
  vtkWriter.addVertexData(forceMap, "force");


  // //////////////////////////////// Read file from WZL ////////////////////////////////
  std::string h5Filename(inName+std::string(".h5"));
  std::cout << "Reading " << h5Filename << " ..." << std::endl;

  H5::H5File h5file(h5Filename, H5F_ACC_RDONLY);

  // Read parameter data
  double param[2];

  H5::DataSet param_dataset(h5file.openDataSet("Werkstoffmodell/E-Modul_Querkontraktion"));
  param_dataset.read(param, H5::PredType::NATIVE_DOUBLE);

  const double E = param[0], nu = param[1];
  std::cout << "   Using: E=" << E << " and nu=" << nu << std::endl;

  // Read solution
  H5::DataSet solution_dataset(h5file.openDataSet("Knoten/Knotenverschiebungen"));

  std::vector<hsize_t> solution_start(3, 0), solution_size(3), solution_count(3);

  // Create file dataspace and get the dimensions of the array stored in the file
  H5::DataSpace dataspace(solution_dataset.getSpace());
  dataspace.getSimpleExtentDims(solution_size.data(), NULL);

  // Set size of slab that will be read into memory
  solution_count[0] = 1;
  solution_count[1] = 1;
  solution_count[2] = solution_size[2];

  // Create dataspace for data in memory
  H5::DataSpace memspace(3, solution_count.data());
  memspace.selectHyperslab(H5S_SELECT_SET, solution_count.data(), solution_start.data());

  std::vector<std::vector<std::vector<double> > > wzl(solution_size[0]);

  for (hsize_t& k = solution_start[0]; k < solution_size[0]; ++k) {
    wzl[k].resize(solution_size[1]);

    for (hsize_t& l = solution_start[1] = 0; l < solution_size[1]; ++l) {
      wzl[k][l].resize(solution_size[2]);

      dataspace.selectHyperslab(H5S_SELECT_SET, solution_count.data(), solution_start.data());
      solution_dataset.read(wzl[k][l].data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);
      
      vtkWriter.addVertexData(wzl[k][l], "wzl_solution_"+toString(k)+"["+toString(l)+"]");
    }
  }

  h5file.close();

  std::cout << "Done" << std::endl;


  // //////////////////////////////// Assemble problem ////////////////////////////////
  std::cout << "Now setting up the stiffness matrix ..." << std::endl;
  watch.start();

  // convert from nu, E to lame constants
  const double
    lambda = (nu*E)/((1+nu)*(1-2*nu)),
    mu = E/(2*(1+nu));

  // create the model describing our problem
  Model model(grid->leafGridView(), rigidNodeMap, lambda, mu);

  // setup grid function space
  FEM fem(grid->leafGridView());
  GFS gfs(grid->leafGridView(), fem);

  // create constraints container
  C cg;
  cg.clear();
  PDELab::constraints(model, gfs, cg);

  // make grid operator
  PDELab::LinearElasticity<Model> lop(model);
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

  //// Create a solver
  std::cout << "   Calling UMFPack constructor ... " << std::endl;
  watch.start();
  UMFPack<ISTL_M> solver(stiffnessMatrix);
  watch.stop();
  std::cout << "   Calling the UMFPack constructor took " << watch.lastElapsed() << " seconds." << std::endl;
  watch.start();

  ///// Apply the solver
  std::vector<std::shared_ptr<V> > x(solution_size[0]);
  ISTL_V rhs(V(gfs).base());

  // Object storing some statistics about the solving process
  InverseOperatorResult statistics;

  for (size_t k = 0; k < forces.size(); ++k) {
    x[k] = std::shared_ptr<V>(new V(gfs, 0));

    rhs = 0;
    rhs[forces[k].first] = forces[k].second;

    solver.apply(*x[k], rhs, statistics);

    gfs.name("dune_solution_" + toString(k));
    PDELab::addSolutionToVTKWriter(vtkWriter, gfs, *x[k]);
  }

  vtkWriter.write(outName);

  // Done.
  watch.stop();
  std::cout << "Getting the solutions from decomposed matrix and writing the VTU file took " << watch.lastElapsed() << " seconds." << std::endl;
  std::cout << "Total time: " << watch.elapsed() << " seconds." << std::endl;
 }
 catch (Exception &e){
   std::cerr << "Error: " << e << std::endl;
   return 1;
 }
