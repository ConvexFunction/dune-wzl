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
#include<dune/grid/yaspgrid.hh>
#include<dune/grid/io/file/vtk/vtkwriter.hh>
#include<dune/istl/bvector.hh>
#include<dune/istl/operators.hh>
#include<dune/istl/solvers.hh>
#include<dune/istl/preconditioners.hh>
#include<dune/istl/io.hh>

#include<dune/pdelab/finiteelementmap/q1fem.hh>
#include<dune/pdelab/constraints/conforming.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspace.hh>
#include<dune/pdelab/gridfunctionspace/gridfunctionspaceutilities.hh>
#include<dune/pdelab/gridfunctionspace/interpolate.hh>
#include<dune/pdelab/constraints/common/constraints.hh>
#include<dune/pdelab/common/function.hh>
#include<dune/pdelab/backend/istlvectorbackend.hh>
#include<dune/pdelab/backend/istlmatrixbackend.hh>
#include<dune/pdelab/backend/istlsolverbackend.hh>
#include<dune/pdelab/localoperator/laplacedirichletp12d.hh>
#include<dune/pdelab/localoperator/poisson.hh>
#include<dune/pdelab/constraints/constraintsparameters.hh>
#include<dune/pdelab/gridoperator/gridoperator.hh>

#include <dune/grid/uggrid.hh>
//#include <dune/grid/alugrid.hh>
#include <dune/istl/umfpack.hh>
#include "nasreader.hh"


using namespace Dune;

//===============================================================
//===============================================================
// Solve the Poisson equation
//           - \Delta u = f in \Omega, 
//                    u = g on \partial\Omega_D
//  -\nabla u \cdot \nu = j on \partial\Omega_N
//===============================================================
//===============================================================


// function for defining the source term
// { constant_function_begin }
template<typename GV, typename RF>
class ConstantFunction
  : public PDELab::AnalyticGridFunctionBase<PDELab::AnalyticGridFunctionTraits<GV,RF,1>,
                                                  ConstantFunction<GV,RF> >
{
public:
  typedef PDELab::AnalyticGridFunctionTraits<GV,RF,1> Traits;
  typedef PDELab::AnalyticGridFunctionBase<Traits,ConstantFunction<GV,RF> > BaseT;

  ConstantFunction (const GV& gv, double value) 
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
// { constant_function_end }


//===============================================================
// Define parameter functions f,g,j and \partial\Omega_D/N
//===============================================================

// constraints parameter class for selecting boundary condition type 
// { dirichlet_boundary_begin }

template<typename GridView>
struct AllDirichlet
  : public PDELab::DirichletConstraintsParameters /*@\label{bcp:base}@*/
{
  const std::vector<bool>& boundaryMap;
  const GridView& gridView;

  AllDirichlet(const std::vector<bool>& map, const GridView& gv) : boundaryMap(map), gridView(gv) {}

  template<typename Intersection>
  bool isDirichlet(const Intersection& intersection,   /*@\label{bcp:name}@*/
                   const FieldVector<typename Intersection::ctype, Intersection::dimension-1> & coord
                  ) const
  {
    bool isBoundary = true;

    for (size_t i = 0; isBoundary && i < 4; ++i) {
      const ReferenceElement<double, 3>& referenceElement = ReferenceElements<double, 3>::general(intersection.inside()->type());

      unsigned localIdx = referenceElement.subEntity(intersection.indexInInside(), 1, i, 3);
      unsigned globalIdx = gridView.indexSet().subIndex(*intersection.inside(), localIdx, 3);

      isBoundary = isBoundary && boundaryMap[globalIdx];
    }

    return isBoundary;
  }
};
// { dirichlet_boundary_end }


// Assemble the Poisson system
// { assembler_signature_begin }
template<typename GridView> 
void poisson( const GridView& gridView, 
              BCRSMatrix<FieldMatrix<double,1,1> >& stiffnessMatrix,
              BlockVector<FieldVector<double,1> >& rhs,
              const std::vector<bool>& boundaryMap)
// { assembler_signature_end }
{
    // make finite element map
    // { assembler_types_begin }
    typedef typename GridView::ctype DF;
    typedef double R;
    // { assembler_types_end }

    // { assembler_fespace_begin }
    // constants and types
    typedef PDELab::ConformingDirichletConstraints Constraints;
    Constraints con;

    // make grid function space
    typedef PDELab::Q1LocalFiniteElementMap<DF,R,GridView::dimension> FEM;
    FEM fem;
    typedef PDELab::ISTLVectorBackend<> VBE;
    typedef PDELab::GridFunctionSpace<GridView,FEM,Constraints,VBE> GFS; 
    GFS gfs(gridView, fem, con);

    // make constraints map and initialize it from a function
    typedef typename GFS::template ConstraintsContainer<R>::Type C;
    C cg;
    cg.clear();

    AllDirichlet<GridView> bctype(boundaryMap, gridView);
    PDELab::constraints(bctype, gfs, cg);
    // { assembler_fespace_end }

    // make grid operator
    // { assembler_create_begin }
    typedef ConstantFunction<GridView,R> FType;
    FType f(gridView, 1);
    typedef ConstantFunction<GridView,R> JType;
    JType j(gridView, 0);
  
    const int qorder = 2;
    typedef PDELab::Poisson<FType,AllDirichlet<GridView>,JType,qorder> LOP; 
    LOP lop(f, bctype, j);

    typedef PDELab::GridOperator<GFS,GFS,LOP,PDELab::ISTLMatrixBackend,R,R,R,C,C> GO;
    GO go(gfs, cg, gfs, cg, lop);
    // { assembler_create_end }

    // { assembler_matrix_begin }
    // make coefficent vector and initialize it from a function
    typedef typename GO::Traits::Domain VectorType;
    VectorType x0(gfs);
    x0 = 0.0;

    // represent operator as a matrix
    typedef typename GO::Jacobian M;
    M m(go);
    m = 0.0;

    go.jacobian(x0,m);
    stiffnessMatrix = m.base();
    // { assembler_matrix_end }

    // evaluate residual w.r.t initial guess
    // { assembler_vector_begin }
    ConstantFunction<GridView,R> g(gridView,0);
    PDELab::interpolate(g, gfs, x0);
    PDELab::set_shifted_dofs(cg, 0.0, x0);

    VectorType r(gfs);
    r = 0.0;

    go.residual(x0, r);
  
    rhs = r.base();
    // { assembler_vector_end }
}
// { assembler_end }



int main(int argc, char** argv) try
{
    // ////////////////////////////////
    //   Generate the grid
    // ////////////////////////////////

    if (argc < 2) {
      std::cout << "dune-wzl in-file out-file" << std::endl;

      return 1;
    }

    const int dim = 3;
    typedef UGGrid<dim> GridType;
    //typedef ALUCubeGrid<dim, dim> GridType;
    typedef FieldVector<double, dim> GlobalVector;
    GridType* grid;
    std::vector<unsigned> rigidNodes;
    std::vector<std::pair<unsigned, GlobalVector> > forces;
    NasReader<double, GridType>::read(argv[1], grid, rigidNodes, forces);
    std::cout << "Forces: " << forces.size() << std::endl;

    // Convert nodes to bitmap
    std::vector<bool> rigidNodeMap(grid->size(dim));
    for (int i = 0; i < grid->size(dim); ++i)
      rigidNodeMap[i] = 0;

    for (int i = 0; i < rigidNodes.size(); ++i)
      rigidNodeMap[rigidNodes[i]] = true;

    std::vector<bool> forceMap(grid->size(dim));
    for (int i = 0; i < grid->size(dim); ++i)
      forceMap[i] = 0;

    for (int i = 0; i < rigidNodes.size(); ++i)
      forceMap[forces[i].first] = true;


    // assemble problem
    typedef BlockVector<FieldVector<double,1> > VectorType;
    typedef BCRSMatrix<FieldMatrix<double,1,1> > MatrixType;


    MatrixType stiffnessMatrix;
    VectorType rhs;

    poisson(grid->leafView(), stiffnessMatrix, rhs, rigidNodeMap);

    // /////////////////////////
    //   Compute solution
    // /////////////////////////
   
    VectorType x(rhs.size());
    x = 0;

    // Object storing some statistics about the solving process
    InverseOperatorResult statistics;

    // Solve!
    UMFPack<MatrixType> solver(stiffnessMatrix, 1); // "1" for verbose output

    solver.apply(x, rhs, statistics);


    // Output
    VTKWriter<GridType::LeafGridView> vtkWriter(grid->leafView());
    vtkWriter.addVertexData(rigidNodeMap, "rigid");
    vtkWriter.addVertexData(forceMap, "force");
    vtkWriter.addVertexData(x, "solution");
    vtkWriter.write(argv[2]);
 }
 catch (Exception &e){
   std::cerr << "Dune reported error: " << e << std::endl;
   return 1;
 }
