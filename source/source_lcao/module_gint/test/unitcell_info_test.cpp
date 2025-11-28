#include "../unitcell_info.h"
#include "gtest/gtest.h"
#include <cmath>

/************************************************
 *  unit test of UnitCellInfo class
 ***********************************************/

/**
 * Tested functions of class UnitCellInfo:
 *  - constructor
 *  - getter functions: get_nbx, get_nby, get_nbz, get_bgrids_num
 *  - getter functions: get_nmx, get_nmy, get_nmz, get_mgrids_num
 *  - bgrid_idx_1Dto3D, bgrid_idx_3Dto1D: biggrid index conversion
 *  - get_bgrid_coord: get Cartesian coordinate of biggrid
 *  - get_bgrid_idx_3d: get 3D index of biggrid from Cartesian coordinate
 *  - get_relative_coord: get relative coordinates between two biggrids
 *  - get_unitcell_idx: get extended unitcell index
 *  - map_ext_idx_to_ucell: map extended index to unitcell
 *  - mgrid_idx_1Dto3D, mgrid_idx_3Dto1D: meshgrid index conversion
 *  - get_mgrid_coord: get Cartesian coordinate of meshgrid
 */

class UnitCellInfoTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple cubic unitcell with vectors (12,0,0), (0,12,0), (0,0,12)
        // with 3x3x3 biggrids and 6x6x6 meshgrids
        unitcell_vec1 = ModuleGint::Vec3d(12.0, 0.0, 0.0);
        unitcell_vec2 = ModuleGint::Vec3d(0.0, 12.0, 0.0);
        unitcell_vec3 = ModuleGint::Vec3d(0.0, 0.0, 12.0);
        nbx = 3;
        nby = 3;
        nbz = 3;
        nmx = 6;
        nmy = 6;
        nmz = 6;
        
        unitcell_info = std::make_shared<ModuleGint::UnitCellInfo>(
            unitcell_vec1, unitcell_vec2, unitcell_vec3,
            nbx, nby, nbz, nmx, nmy, nmz);
    }

    ModuleGint::Vec3d unitcell_vec1, unitcell_vec2, unitcell_vec3;
    int nbx, nby, nbz;
    int nmx, nmy, nmz;
    std::shared_ptr<ModuleGint::UnitCellInfo> unitcell_info;
};

// Test constructor and getter functions
TEST_F(UnitCellInfoTest, Constructor_BigGridDimensions)
{
    EXPECT_EQ(unitcell_info->get_nbx(), nbx);
    EXPECT_EQ(unitcell_info->get_nby(), nby);
    EXPECT_EQ(unitcell_info->get_nbz(), nbz);
    EXPECT_EQ(unitcell_info->get_bgrids_num(), nbx * nby * nbz);
}

TEST_F(UnitCellInfoTest, Constructor_MeshGridDimensions)
{
    EXPECT_EQ(unitcell_info->get_nmx(), nmx);
    EXPECT_EQ(unitcell_info->get_nmy(), nmy);
    EXPECT_EQ(unitcell_info->get_nmz(), nmz);
    EXPECT_EQ(unitcell_info->get_mgrids_num(), nmx * nmy * nmz);
}

TEST_F(UnitCellInfoTest, Constructor_BigGridInfo)
{
    auto bgrid_info = unitcell_info->get_bgrid_info();
    EXPECT_NE(bgrid_info, nullptr);
    // Each biggrid has (nmx/nbx) x (nmy/nby) x (nmz/nbz) = 2x2x2 meshgrids
    EXPECT_EQ(bgrid_info->get_nmx(), 2);
    EXPECT_EQ(bgrid_info->get_nmy(), 2);
    EXPECT_EQ(bgrid_info->get_nmz(), 2);
}

// Test biggrid index conversion
TEST_F(UnitCellInfoTest, BgridIdx_1Dto3D_Origin)
{
    ModuleGint::Vec3i result = unitcell_info->bgrid_idx_1Dto3D(0);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(UnitCellInfoTest, BgridIdx_3Dto1D_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    EXPECT_EQ(unitcell_info->bgrid_idx_3Dto1D(index_3d), 0);
}

TEST_F(UnitCellInfoTest, BgridIdx_Conversion_Roundtrip)
{
    for (int i = 0; i < nbx * nby * nbz; ++i)
    {
        ModuleGint::Vec3i idx_3d = unitcell_info->bgrid_idx_1Dto3D(i);
        EXPECT_EQ(unitcell_info->bgrid_idx_3Dto1D(idx_3d), i);
    }
}

// Test get_bgrid_coord with Vec3i
TEST_F(UnitCellInfoTest, GetBgridCoord_Vec3i_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    ModuleGint::Vec3d coord = unitcell_info->get_bgrid_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

TEST_F(UnitCellInfoTest, GetBgridCoord_Vec3i_Unit)
{
    // Biggrid size = unitcell_vec / nb = (4, 4, 4)
    ModuleGint::Vec3i index_3d(1, 0, 0);
    ModuleGint::Vec3d coord = unitcell_info->get_bgrid_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 4.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
    
    index_3d = ModuleGint::Vec3i(0, 2, 0);
    coord = unitcell_info->get_bgrid_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 8.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

// Test get_bgrid_coord with int (1D index)
TEST_F(UnitCellInfoTest, GetBgridCoord_1D)
{
    // Index 0 should be origin
    ModuleGint::Vec3d coord = unitcell_info->get_bgrid_coord(0);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

// Test get_bgrid_idx_3d
TEST_F(UnitCellInfoTest, GetBgridIdx3D_Origin)
{
    ModuleGint::Vec3d coord(0.0, 0.0, 0.0);
    ModuleGint::Vec3i idx = unitcell_info->get_bgrid_idx_3d(coord);
    EXPECT_EQ(idx.x, 0);
    EXPECT_EQ(idx.y, 0);
    EXPECT_EQ(idx.z, 0);
}

TEST_F(UnitCellInfoTest, GetBgridIdx3D_InFirstBgrid)
{
    // Point inside the first biggrid (size 4x4x4)
    ModuleGint::Vec3d coord(2.0, 3.0, 1.0);
    ModuleGint::Vec3i idx = unitcell_info->get_bgrid_idx_3d(coord);
    EXPECT_EQ(idx.x, 0);
    EXPECT_EQ(idx.y, 0);
    EXPECT_EQ(idx.z, 0);
}

TEST_F(UnitCellInfoTest, GetBgridIdx3D_InSecondBgrid)
{
    ModuleGint::Vec3d coord(5.0, 1.0, 1.0);  // x > 4, so in second biggrid along x
    ModuleGint::Vec3i idx = unitcell_info->get_bgrid_idx_3d(coord);
    EXPECT_EQ(idx.x, 1);
    EXPECT_EQ(idx.y, 0);
    EXPECT_EQ(idx.z, 0);
}

// Test get_relative_coord
TEST_F(UnitCellInfoTest, GetRelativeCoord_Same)
{
    ModuleGint::Vec3i idx_a(1, 1, 1);
    ModuleGint::Vec3i idx_b(1, 1, 1);
    ModuleGint::Vec3d rel_coord = unitcell_info->get_relative_coord(idx_a, idx_b);
    EXPECT_DOUBLE_EQ(rel_coord.x, 0.0);
    EXPECT_DOUBLE_EQ(rel_coord.y, 0.0);
    EXPECT_DOUBLE_EQ(rel_coord.z, 0.0);
}

TEST_F(UnitCellInfoTest, GetRelativeCoord_Different)
{
    ModuleGint::Vec3i idx_a(2, 1, 0);
    ModuleGint::Vec3i idx_b(0, 1, 0);
    ModuleGint::Vec3d rel_coord = unitcell_info->get_relative_coord(idx_a, idx_b);
    // (2-0) * 4 = 8 in x direction
    EXPECT_DOUBLE_EQ(rel_coord.x, 8.0);
    EXPECT_DOUBLE_EQ(rel_coord.y, 0.0);
    EXPECT_DOUBLE_EQ(rel_coord.z, 0.0);
}

// Test get_unitcell_idx
TEST_F(UnitCellInfoTest, GetUnitcellIdx_InFirstUcell)
{
    ModuleGint::Vec3i idx(0, 1, 2);
    ModuleGint::Vec3i ucell_idx = unitcell_info->get_unitcell_idx(idx);
    EXPECT_EQ(ucell_idx.x, 0);
    EXPECT_EQ(ucell_idx.y, 0);
    EXPECT_EQ(ucell_idx.z, 0);
}

TEST_F(UnitCellInfoTest, GetUnitcellIdx_Extended)
{
    ModuleGint::Vec3i idx(3, 4, 5);  // nbx=3, so x=3 is in next unitcell
    ModuleGint::Vec3i ucell_idx = unitcell_info->get_unitcell_idx(idx);
    EXPECT_EQ(ucell_idx.x, 1);
    EXPECT_EQ(ucell_idx.y, 1);
    EXPECT_EQ(ucell_idx.z, 1);
}

TEST_F(UnitCellInfoTest, GetUnitcellIdx_Negative)
{
    ModuleGint::Vec3i idx(-1, -1, -1);
    ModuleGint::Vec3i ucell_idx = unitcell_info->get_unitcell_idx(idx);
    EXPECT_EQ(ucell_idx.x, -1);
    EXPECT_EQ(ucell_idx.y, -1);
    EXPECT_EQ(ucell_idx.z, -1);
}

// Test map_ext_idx_to_ucell
TEST_F(UnitCellInfoTest, MapExtIdxToUcell_InFirstUcell)
{
    ModuleGint::Vec3i idx(1, 1, 1);
    ModuleGint::Vec3i mapped_idx = unitcell_info->map_ext_idx_to_ucell(idx);
    EXPECT_EQ(mapped_idx.x, 1);
    EXPECT_EQ(mapped_idx.y, 1);
    EXPECT_EQ(mapped_idx.z, 1);
}

TEST_F(UnitCellInfoTest, MapExtIdxToUcell_Extended)
{
    ModuleGint::Vec3i idx(4, 5, 6);  // should map to (1, 2, 0)
    ModuleGint::Vec3i mapped_idx = unitcell_info->map_ext_idx_to_ucell(idx);
    EXPECT_EQ(mapped_idx.x, 1);
    EXPECT_EQ(mapped_idx.y, 2);
    EXPECT_EQ(mapped_idx.z, 0);
}

TEST_F(UnitCellInfoTest, MapExtIdxToUcell_Negative)
{
    ModuleGint::Vec3i idx(-1, -1, -1);  // should map to (2, 2, 2)
    ModuleGint::Vec3i mapped_idx = unitcell_info->map_ext_idx_to_ucell(idx);
    EXPECT_EQ(mapped_idx.x, 2);
    EXPECT_EQ(mapped_idx.y, 2);
    EXPECT_EQ(mapped_idx.z, 2);
}

// Test meshgrid index conversion
TEST_F(UnitCellInfoTest, MgridIdx_1Dto3D_Origin)
{
    ModuleGint::Vec3i result = unitcell_info->mgrid_idx_1Dto3D(0);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(UnitCellInfoTest, MgridIdx_3Dto1D_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    EXPECT_EQ(unitcell_info->mgrid_idx_3Dto1D(index_3d), 0);
}

TEST_F(UnitCellInfoTest, MgridIdx_Conversion_Roundtrip)
{
    for (int i = 0; i < std::min(nmx * nmy * nmz, 27); ++i)  // Test subset
    {
        ModuleGint::Vec3i idx_3d = unitcell_info->mgrid_idx_1Dto3D(i);
        EXPECT_EQ(unitcell_info->mgrid_idx_3Dto1D(idx_3d), i);
    }
}

// Test get_mgrid_coord
TEST_F(UnitCellInfoTest, GetMgridCoord_Vec3i_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    ModuleGint::Vec3d coord = unitcell_info->get_mgrid_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

TEST_F(UnitCellInfoTest, GetMgridCoord_Vec3i_Unit)
{
    // Meshgrid size = unitcell_vec / nm = (2, 2, 2)
    ModuleGint::Vec3i index_3d(1, 0, 0);
    ModuleGint::Vec3d coord = unitcell_info->get_mgrid_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 2.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

TEST_F(UnitCellInfoTest, GetMgridCoord_1D)
{
    ModuleGint::Vec3d coord = unitcell_info->get_mgrid_coord(0);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}
