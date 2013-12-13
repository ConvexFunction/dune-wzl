// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"     
#endif

#include<iostream>
#include<vector>
#include<string>

#include<dune/common/exceptions.hh>
#include<dune/common/fvector.hh>
#include<dune/common/timer.hh>

#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk/vtkwriter.hh>
#include<dune/grid/uggrid.hh>
//#include<dune/grid/alugrid.hh>

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
#include <dune/pdelab/gridfunctionspace/vtk.hh>
#include<dune/pdelab/constraints/common/constraints.hh>
#include<dune/pdelab/common/function.hh>
#include<dune/pdelab/backend/istlvectorbackend.hh>
#include<dune/pdelab/backend/istlmatrixbackend.hh>
#include<dune/pdelab/backend/istlsolverbackend.hh>
//#include<dune/pdelab/localoperator/laplacedirichletp12d.hh>
//#include<dune/pdelab/localoperator/poisson.hh>
#include<dune/pdelab/localoperator/linearelasticity.hh>
#include<dune/pdelab/constraints/constraintsparameters.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>

#include "nasreader.hh"


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
  PDELab::ISTLVectorBackend<>,
  PDELab::ISTLVectorBackend<>,
  PDELab::ConformingDirichletConstraints,
  PDELab::LexicographicOrderingTag,
  PDELab::DefaultLeafOrderingTag
  > GFS;

typedef GFS::ConstraintsContainer<Real>::Type C;

typedef PDELab::GridOperator<GFS,GFS,PDELab::LinearElasticity,PDELab::ISTLMatrixBackend,Real,Real,Real,C,C> GO;
typedef GO::Traits::Domain V;
typedef GO::Jacobian M;
typedef M::BaseT ISTL_M;
typedef V::BaseT ISTL_V;


// function for defining the source term
template<typename GV, typename RF>
class ConstantFunction
  : public PDELab::AnalyticGridFunctionBase<PDELab::AnalyticGridFunctionTraits<GV,RF,1>,
                                                  ConstantFunction<GV,RF> >
{
public:
  typedef PDELab::AnalyticGridFunctionTraits<GV,RF,1> Traits;
  typedef PDELab::AnalyticGridFunctionBase<Traits,ConstantFunction<GV,RF> > BaseT;

  ConstantFunction (const GV& gv, Real value) 
  : BaseT(gv), value_(value)
  {}
  
  void evaluateGlobal (const typename Traits::DomainType& x, 
                       typename Traits::RangeType& y) const
  {
    y=value_;
  }
private:
  typename Traits::RangeType value_;
};


//===============================================================
// Define parameter functions f,g,j and \partial\Omega_D/N
//===============================================================

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
void assemble(const GridView& gridView,
              const GFS& gfs,
              ISTL_M& stiffnessMatrix,
              ISTL_V& rhs,
              const std::vector<bool>& boundaryMap)
{
    // make constraints map and initialize it from a function
    C cg;
    cg.clear();

    Dirichlet<GridView> bctype(boundaryMap, gridView);
    PDELab::constraints(bctype, gfs, cg);

    // make grid operator
    ConstantFunction<typename GridType::LeafGridView,Real> f(gridView, 1);

    PDELab::LinearElasticity lop(3e5, 3e5, 0);

    GO go(gfs, cg, gfs, cg, lop);

    // make coefficent vector and initialize it from a function
    V x0(gfs);
    x0 = 0.0;

    ConstantFunction<GridView,Real> g(gridView, 1);
    PDELab::interpolate(g, gfs, x0);
    //PDELab::set_shifted_dofs(cg, 0.0, x0);
    PDELab::set_nonconstrained_dofs(cg, 0.0, x0);

    // represent operator as a matrix
    M m(go);
    m = 0.0;

    go.jacobian(x0, m);

    // evaluate residual w.r.t initial guess
    V r(gfs);
    r = 0.0;

    go.residual(x0, r);

    // get actual matrices
    stiffnessMatrix = m.base();
    rhs = r.base();
}


int main(int argc, char** argv) try
{
    // ////////////////////////////////
    //   Generate the grid
    // ////////////////////////////////

    if (argc < 2) {
      std::cout << "dune-wzl in-file out-file" << std::endl;

      return 1;
    }

    std::cout << "Reading " << argv[1] << " ..." << std::endl;

    // Create and start timer
    Timer watch;

    // Create grid by reading *.nas file
    GridType* grid;

    std::vector<unsigned> rigidNodes;
    std::vector<std::pair<unsigned, GlobalVector> > forces;

    NasReader<Real, GridType>::read(argv[1], grid, rigidNodes, forces);

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
    std::cout << "Now setting up the stiffness matrix and the right hand side ..." << std::endl;
    watch.start();

    // assemble problem
    FEM fem;
    GFS gfs(grid->leafView(), fem);
    gfs.name("displacement");
  
    ISTL_M stiffnessMatrix;
    ISTL_V rhs;

    assemble(grid->leafView(), gfs, stiffnessMatrix, rhs, rigidNodeMap);

    // Output how long setting up the matrices took us
    watch.stop();
    std::cout << "Assembling the stiffness matrix and setting up the right hand side took " << watch.lastElapsed() << " seconds." << std::endl;
    std::cout << "Now computing the solution ..." << std::endl;
    watch.start();

    // /////////////////////////
    //   Compute solution
    // /////////////////////////

    // Object storing some statistics about the solving process
    InverseOperatorResult statistics;

    // Solve!
    /*
    MatrixAdapter<ISTL_M,ISTL_V,ISTL_V> opa(stiffnessMatrix);
    SeqILU0<ISTL_M,ISTL_V,ISTL_V> ilu0(stiffnessMatrix,1e-2);
    CGSolver<ISTL_V> solver(opa,ilu0,1E-4,150,2);
    */

    std::cout << "   Elapsed time before calling UMFPack constructor: "<< watch.elapsed() << std::endl;
    UMFPack<ISTL_M> solver(stiffnessMatrix);
    std::cout << "   Elapsed time after calling UMFPack constructor: " << watch.elapsed() << std::endl;

    V x(gfs, 0);
    solver.apply(x, rhs, statistics);
    std::cout << "   Elapsed time after apply: " << watch.elapsed() << std::endl;

    // Read our stopwatch for a last time
    watch.stop();
    std::cout << "Getting the solution took " << watch.lastElapsed() << " seconds." << std::endl;
    std::cout << "Total time: " << watch.elapsed() << " seconds." << std::endl;

    // Output
    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafView());

    vtkWriter.addVertexData(rigidNodeMap, "rigid");
    vtkWriter.addVertexData(forceMap, "force");
    PDELab::addSolutionToVTKWriter(vtkWriter, gfs, x);

    vtkWriter.write(argv[2]);
 }
 catch (Exception &e){
   std::cerr << "Dune reported error: " << e << std::endl;
   return 1;
 }
