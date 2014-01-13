// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"     
#endif

#include<iostream>
#include<memory>
#include<string>
#include<sstream>
#include<vector>

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

typedef double Real;

typedef BlockVector<FieldVector<Real,dim> > VectorType;
typedef BCRSMatrix<FieldMatrix<Real,dim,dim> > MatrixType;
typedef FieldVector<Real, dim> GlobalVector;

typedef UGGrid<dim> GridType;
//typedef ALUCubeGrid<dim, dim> GridType;

typedef PDELab::Q1LocalFiniteElementMap<Real,Real,dim> FEM;
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

typedef GFS::ConstraintsContainer<Real>::Type C;

typedef PDELab::GridOperator<GFS,GFS,PDELab::LinearElasticity,PDELab::ISTLMatrixBackend,Real,Real,Real,C,C> GO;
typedef GO::Traits::Domain V;
typedef GO::Jacobian M;
typedef M::BaseT ISTL_M;
typedef V::BaseT ISTL_V;

// constraints parameter class for selecting boundary condition type 
template<typename GridView>
struct Dirichlet
  : public PDELab::DirichletConstraintsParameters /*@\label{bcp:base}@*/
{
  const std::vector<bool>& boundaryMap;
  const GridView& gridView;

  Dirichlet(const std::vector<bool>& map, const GridView& gv) : boundaryMap(map), gridView(gv) {}

  template<typename Intersection>
  bool isDirichlet(const Intersection& intersection,   /*@\label{bcp:name}@*/
                   const FieldVector<typename Intersection::ctype, Intersection::dimension-1> & coord
                  ) const
  {
    bool isBoundary = true;

    for (size_t i = 0; isBoundary && i < 4; ++i) {
      const ReferenceElement<Real, dim>& referenceElement = ReferenceElements<Real, dim>::general(intersection.inside()->type());

      unsigned localIdx = referenceElement.subEntity(intersection.indexInInside(), 1, i, dim);
      unsigned globalIdx = gridView.indexSet().subIndex(*intersection.inside(), localIdx, dim);

      isBoundary = isBoundary && boundaryMap[globalIdx];
    }

    return isBoundary;
  }
};


// Assemble the system
template<typename GridView> 
void assembleMatrix(const GridView& gridView,
                  const Real& E, const Real& nu,
                  const GFS& gfs,
                  ISTL_M& stiffnessMatrix,
                  const std::vector<bool>& boundaryMap)
{
    // make constraints map and initialize it via a Dirichlet<GridView> object
    C cg;
    cg.clear();

    Dirichlet<GridView> bctype(boundaryMap, gridView);
    PDELab::constraints(bctype, gfs, cg);

    // convert from nu, E to lame constants
    const Real
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
    stiffnessMatrix = m.base();
}

template<typename T>
std::string toString(const T& t) {
  return static_cast<std::ostringstream*>( &(std::ostringstream() << t) )->str();
}

int main(int argc, char** argv) try
{
    // ////////////////////////////////
    //   Generate the grid
    // ////////////////////////////////

    if (argc < 2) {
      std::cout << "dune-wzl in-file out-file (without file extensions)" << std::endl;

      return 1;
    }


    //// Read the nas file
    std::string nasFilename(argv[1]+std::string(".nas"));
    std::cout << "Reading " << nasFilename << " ..." << std::endl;

    // Create and start timer
    Timer watch;

    // Create grid by reading *.nas file
    GridType* grid;

    std::vector<unsigned> rigidNodes;
    std::vector<std::pair<unsigned, GlobalVector> > forces;

    NasReader<Real, GridType>::read(nasFilename, grid, rigidNodes, forces);

    // Convert nodes to bitmaps
    std::vector<bool> rigidNodeMap(grid->size(dim));
    for (int i = 0; i < grid->size(dim); ++i)
      rigidNodeMap[i] = 0;

    for (int i = 0; i < rigidNodes.size(); ++i)
      rigidNodeMap[rigidNodes[i]] = true;

    std::vector<bool> forceMap(grid->size(dim));
    for (int i = 0; i < grid->size(dim); ++i)
      forceMap[i] = 0;

    for (int i = 0; i < forces.size(); ++i)
      forceMap[forces[i].first] = true;

    // Output how long reading the file took us
    watch.stop();
    std::cout << "Reading and setting up vectors took " << watch.lastElapsed() << " seconds." << std::endl;

    // Prepare vtkWriter
    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafView());

    vtkWriter.addVertexData(rigidNodeMap, "rigid");
    vtkWriter.addVertexData(forceMap, "force");

    //// Read file from WZL
    std::string h5Filename(argv[1]+std::string(".h5"));
    std::cout << "Reading " << h5Filename << " ..." << std::endl;

    H5::H5File h5file(h5Filename, H5F_ACC_RDONLY);

    // Read parameter data
    Real param[2];
    h5file.openDataSet("Werkstoffmodell/E-Modul_Querkontraktion").read(param, H5::PredType::NATIVE_DOUBLE);
     std::cout << "   Using: E=" << param[0] << " and nu=" << param[1] << std::endl;

    // Read solution
    H5::DataSet solution_dataset = h5file.openDataSet("Knoten/Knotenverschiebungen");

    std::vector<hsize_t> solution_start(3, 0), solution_size(3), solution_count(3);

    // I don't get how DataSpaces are supposed to work.

    // H5::DataSpace memspace(solution_dataset.getSpace());
    H5::DataSpace dataspace(solution_dataset.getSpace());
    dataspace.getSimpleExtentDims(solution_size.data(), NULL);
    // solution_count[0] = 1; solution_count[1] = 1; solution_count[2] = solution_size[2];
    // memspace.selectHyperslab(H5S_SELECT_SET, solution_count.data(), solution_start.data());

    // Workaround: Just pull out all the data.
    std::vector<Real> full_wzl(solution_size[0] * solution_size[1] * solution_size[2]);
    solution_dataset.read(full_wzl.data(), H5::PredType::NATIVE_DOUBLE); //, memspace, dataspace);

    std::vector<std::vector<std::vector<Real> > > wzl(solution_size[0]);

    for (size_t k = 0; k < solution_size[0]; ++k, ++solution_start[0]) {
      wzl[k].resize(solution_size[1]);

      for (size_t l = 0; l < solution_size[1]; ++l, ++solution_start[1]) {
        // dataspace.selectHyperslab(H5S_SELECT_SET, solution_count.data(), solution_start.data());
        wzl[k][l] = std::vector<Real>(&full_wzl[static_cast<size_t>((k*solution_size[1]+l)    *solution_size[2])],
                                      &full_wzl[static_cast<size_t>((k*solution_size[1]+(l+1))*solution_size[2])]);

        vtkWriter.addVertexData(wzl[k][l], "wzl_solution_" + toString(k) + "_[" + toString(l) + "]");
      }
    }

    full_wzl.resize(0);

    h5file.close();

    std::cout << "Done" << std::endl;

    // Assemble problem
    std::cout << "Now setting up the stiffness matrix ..." << std::endl;
    watch.start();

    FEM fem;
    GFS gfs(grid->leafView(), fem);

    ISTL_M stiffnessMatrix;
    assembleMatrix(grid->leafView(), param[0], param[1], gfs, stiffnessMatrix, rigidNodeMap);

    // Output how long setting up the matrices took us
    watch.stop();
    std::cout << "Assembling the stiffness matrix took " << watch.lastElapsed() << " seconds." << std::endl;

    // //////////////es///////////
    //   Compute solution
    // /////////////////////////
    
    std::cout << "Now computing the solutions ..." << std::endl;
    watch.start();

    // Solve!

    // Create a solver
    std::cout << "   Elapsed time before calling UMFPack constructor: "<< watch.elapsed() << std::endl;
    UMFPack<ISTL_M> solver(stiffnessMatrix);
    std::cout << "   Elapsed time after calling UMFPack constructor: " << watch.elapsed() << std::endl;

    // Object storing some statistics about the solving process
    InverseOperatorResult statistics;

    // Apply the solver
    std::vector<std::shared_ptr<V> > x(forces.size());
    ISTL_V rhs(V(gfs).base());

    for (size_t k = 0; k < forces.size(); ++k) {
      x[k] = std::shared_ptr<V>(new V(gfs, 0));

      gfs.name("solution_" + toString(k));

      rhs = 0;
      rhs[forces[k].first] = forces[k].second;

      solver.apply(*(x[k]), rhs, statistics);

      // Test: Calculate rhs -= A*x
      stiffnessMatrix.mmv((*x[k]).base(), rhs);
      std::cout << "   Test " << k << ": " << sqrt(rhs*rhs) << std::endl;

      PDELab::addSolutionToVTKWriter(vtkWriter, gfs, *(x[k]));
    }

    // Done.
    watch.stop();
    std::cout << "Getting the solutions took " << watch.lastElapsed() << " seconds." << std::endl;
    std::cout << "Total time: " << watch.elapsed() << " seconds." << std::endl;

    //// Output
    vtkWriter.write(argv[2]);
 }
 catch (Exception &e){
   std::cerr << "Error: " << e << std::endl;
   return 1;
 }
