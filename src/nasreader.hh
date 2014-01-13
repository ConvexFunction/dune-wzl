// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef NASREADER_HH
#define NASREADER_HH

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include <dune/common/exceptions.hh>
#include <dune/common/fvector.hh>

#include <dune/grid/common/boundarysegment.hh>
#include <dune/grid/common/gridfactory.hh>
#include <dune/grid/common/scsgmapper.hh>

#include <dune/geometry/type.hh>


// DOCUMENT

template<typename ctype, typename GridType>
class NasReader
{
private:
  static std::string readKeyword(std::istream& file) {
    // if we are not at the beginning of the file skip to the next line
    if (file.tellg() != file.beg) {
      std::string dummy;
      getline(file, dummy);
    }

    // extract keyword on current line, if nothing went wrong
    std::string keyword;

    if (file.good())
      file >> keyword;
    else
      DUNE_THROW(Dune::IOError, "File is not good.");

    return keyword;
  }

public:
  // static data
  static const int dimWorld = GridType::dimensionworld;
  dune_static_assert( (dimWorld == 3), "NasReader requires dimWorld == 3." );

  // typedefs
  typedef Dune::FieldVector<ctype, dimWorld> GlobalVector;
  typedef GridType Grid;

  // DOCME
  static void read (const std::string& fileName, Grid*& grid, std::vector<unsigned>& rigid, std::vector<std::pair<unsigned, GlobalVector> >& force) {
    std::ifstream file(fileName.c_str());

    if (!file.is_open())
      DUNE_THROW(Dune::IOError, "Could not open " << fileName << " with read access.");


    // Create a factory for our grid.
    Dune::GridFactory<Grid> factory;

    // We assume everything is in ascending order, i.e. first node comes first etc. and "element numbers" etc. are ignored.
    std::string keyword;
    bool inCOMPBlock = false;

    do {
      std::string _;
      keyword = readKeyword(file);

      if (!keyword.compare("GRID*")) {
        GlobalVector position;

        // GRID* "node number" "x" "y"
        // * "z"
        file >> _ >> position[0] >> position[1]
             >> _ >> position[2];

        factory.insertVertex(position);
      } else if (!keyword.compare("CHEXA")) {
        Dune::GeometryType type(Dune::GeometryType::cube, 3);
        std::vector<unsigned> vertices(8);

        // CHEXA "element number" "element group" "node 1" "node 2" "node 3" "node 4" "node 5" "node 6""+" "element number"
        // + "element number" "node 7" "node 8"
        // Careful: The order of the nodes of hexahedrons in DUNE is different from what appears to be the standard order
        file >> _ >> _ >> vertices[0] >> vertices[1] >> vertices[3] >> vertices[2] >> vertices[4] >> vertices[5] >> _ >> _
             >> _ >> _ >> vertices[7] >> vertices[6];

        // Node numbers start by zero in DUNE
        for (int i = 0; i < 8; ++i)
          vertices[i] -= 1;

        factory.insertElement(type, vertices);
      } else if (!keyword.compare("$HMNAME")) {
        file >> _;

        // Only if we are in the COMP block, the lines starting with SPC contain the rigid nodes
        if (!_.compare("COMP"))
          inCOMPBlock = true;
        else
          inCOMPBlock = false;
      } else if (!keyword.compare("SPC")) {
        // Only if we are in the COMP block, the lines starting with SPC contain the rigid nodes
        if (inCOMPBlock) {
          unsigned rigidNode;

          // SPC "identifier" "number of rigid node" "bound degrees of freedom" "'unrigidness'"
          file >> _ >> rigidNode >> _ >> _;

          // Node numbers start by zero in DUNE
          rigid.push_back(rigidNode-1);
        }
      } else if (!keyword.compare("FORCE*")) {
        GlobalVector forceVec;
        unsigned node;

        // FORCE* "case number" "node" "unknown" "magnitude"
        // * "x" "y" "z"
        file >> _ >> node >> _ >> _
             >> _ >> forceVec[0] >> forceVec[1] >> forceVec[2];

        // Node numbers start by zero in DUNE
        force.push_back(std::make_pair(node-1, forceVec));
      }
    } while (keyword.compare("ENDDATA"));

    // we are done with the file
    file.close();

    // finally create the grid
    grid = factory.createGrid();
  }
};

#endif
