#include "../localcell_info.h"
#include "gtest/gtest.h"
#include <cmath>

/************************************************
 *  unit test of LocalCellInfo class
 ***********************************************/

/**
 * Tested functions of class LocalCellInfo:
 *  - constructor
 *  - getter functions for local cell dimensions
 *  - bgrid_idx_3Dto1D, bgrid_idx_1Dto3D: local biggrid index conversion
 *  - get_bgrid_global_idx_3D: get global 3D index
 *  - get_bgrid_global_idx_1D: get global 1D index
 *  - get_bgrid_local_idx_3D: get local 3D index
 *  - get_bgrid_local_idx_1D: get local 1D index
 *  - get_bgrid_global_coord_3D: get global coordinate
 *  - is_bgrid_in_lcell: check if biggrid is in local cell
 *  - mgrid_idx_3Dto1D, mgrid_idx_1Dto3D: local meshgrid index conversion
 *  - get_mgrid_global_idx_3D: get global 3D index of meshgrid
 *  - get_mgrid_global_idx_1D: get global 1D index of meshgrid
 */

class LocalCellInfoTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // Create a unitcell with 4x4x4 biggrids and 8x8x8 meshgrids
        unitcell_vec1 = ModuleGint::Vec3d(16.0, 0.0, 0.0);
        unitcell_vec2 = ModuleGint::Vec3d(0.0, 16.0, 0.0);
        unitcell_vec3 = ModuleGint::Vec3d(0.0, 0.0, 16.0);
        nbx_global = 4;
        nby_global = 4;
        nbz_global = 4;
        nmx_global = 8;
        nmy_global = 8;
        nmz_global = 8;
        
        unitcell_info = std::make_shared<ModuleGint::UnitCellInfo>(
            unitcell_vec1, unitcell_vec2, unitcell_vec3,
            nbx_global, nby_global, nbz_global,
            nmx_global, nmy_global, nmz_global);

        // Create a local cell that covers part of the unitcell
        // Starting at (1, 1, 1) with 2x2x2 biggrids
        startidx_bx = 1;
        startidx_by = 1;
        startidx_bz = 1;
        nbx_local = 2;
        nby_local = 2;
        nbz_local = 2;
        
        localcell_info = std::make_shared<ModuleGint::LocalCellInfo>(
            startidx_bx, startidx_by, startidx_bz,
            nbx_local, nby_local, nbz_local,
            unitcell_info);
    }

    ModuleGint::Vec3d unitcell_vec1, unitcell_vec2, unitcell_vec3;
    int nbx_global, nby_global, nbz_global;
    int nmx_global, nmy_global, nmz_global;
    std::shared_ptr<ModuleGint::UnitCellInfo> unitcell_info;
    
    int startidx_bx, startidx_by, startidx_bz;
    int nbx_local, nby_local, nbz_local;
    std::shared_ptr<ModuleGint::LocalCellInfo> localcell_info;
};

// Test constructor and getter functions
TEST_F(LocalCellInfoTest, Constructor_BigGridDimensions)
{
    EXPECT_EQ(localcell_info->get_startidx_bx(), startidx_bx);
    EXPECT_EQ(localcell_info->get_startidx_by(), startidx_by);
    EXPECT_EQ(localcell_info->get_startidx_bz(), startidx_bz);
    EXPECT_EQ(localcell_info->get_nbx(), nbx_local);
    EXPECT_EQ(localcell_info->get_nby(), nby_local);
    EXPECT_EQ(localcell_info->get_nbz(), nbz_local);
    EXPECT_EQ(localcell_info->get_bgrids_num(), nbx_local * nby_local * nbz_local);
}

TEST_F(LocalCellInfoTest, Constructor_MeshGridDimensions)
{
    // Each biggrid has 2x2x2 meshgrids (nmx/nbx = 8/4 = 2)
    // Local cell has 2x2x2 biggrids
    // So local cell has 4x4x4 meshgrids
    EXPECT_EQ(localcell_info->get_mgrids_num(), 4 * 4 * 4);
}

TEST_F(LocalCellInfoTest, Constructor_UnitCellInfo)
{
    EXPECT_EQ(localcell_info->get_unitcell_info(), unitcell_info);
    EXPECT_NE(localcell_info->get_bgrid_info(), nullptr);
}

// Test local biggrid index conversion
TEST_F(LocalCellInfoTest, BgridIdx_1Dto3D_Origin)
{
    ModuleGint::Vec3i result = localcell_info->bgrid_idx_1Dto3D(0);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(LocalCellInfoTest, BgridIdx_3Dto1D_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    EXPECT_EQ(localcell_info->bgrid_idx_3Dto1D(index_3d), 0);
}

TEST_F(LocalCellInfoTest, BgridIdx_Conversion_Roundtrip)
{
    for (int i = 0; i < nbx_local * nby_local * nbz_local; ++i)
    {
        ModuleGint::Vec3i idx_3d = localcell_info->bgrid_idx_1Dto3D(i);
        EXPECT_EQ(localcell_info->bgrid_idx_3Dto1D(idx_3d), i);
    }
}

// Test get_bgrid_global_idx_3D with Vec3i input
TEST_F(LocalCellInfoTest, GetBgridGlobalIdx3D_Vec3i_Origin)
{
    ModuleGint::Vec3i local_idx(0, 0, 0);
    ModuleGint::Vec3i global_idx = localcell_info->get_bgrid_global_idx_3D(local_idx);
    EXPECT_EQ(global_idx.x, startidx_bx);
    EXPECT_EQ(global_idx.y, startidx_by);
    EXPECT_EQ(global_idx.z, startidx_bz);
}

TEST_F(LocalCellInfoTest, GetBgridGlobalIdx3D_Vec3i_Offset)
{
    ModuleGint::Vec3i local_idx(1, 1, 1);
    ModuleGint::Vec3i global_idx = localcell_info->get_bgrid_global_idx_3D(local_idx);
    EXPECT_EQ(global_idx.x, startidx_bx + 1);
    EXPECT_EQ(global_idx.y, startidx_by + 1);
    EXPECT_EQ(global_idx.z, startidx_bz + 1);
}

// Test get_bgrid_global_idx_3D with int input
TEST_F(LocalCellInfoTest, GetBgridGlobalIdx3D_Int_Origin)
{
    ModuleGint::Vec3i global_idx = localcell_info->get_bgrid_global_idx_3D(0);
    EXPECT_EQ(global_idx.x, startidx_bx);
    EXPECT_EQ(global_idx.y, startidx_by);
    EXPECT_EQ(global_idx.z, startidx_bz);
}

// Test get_bgrid_global_idx_1D
TEST_F(LocalCellInfoTest, GetBgridGlobalIdx1D)
{
    // Local index 0 corresponds to global (1,1,1)
    int global_idx = localcell_info->get_bgrid_global_idx_1D(0);
    ModuleGint::Vec3i expected_3d(startidx_bx, startidx_by, startidx_bz);
    int expected_1d = unitcell_info->bgrid_idx_3Dto1D(expected_3d);
    EXPECT_EQ(global_idx, expected_1d);
}

// Test get_bgrid_local_idx_3D
TEST_F(LocalCellInfoTest, GetBgridLocalIdx3D)
{
    ModuleGint::Vec3i global_idx(startidx_bx, startidx_by, startidx_bz);
    ModuleGint::Vec3i local_idx = localcell_info->get_bgrid_local_idx_3D(global_idx);
    EXPECT_EQ(local_idx.x, 0);
    EXPECT_EQ(local_idx.y, 0);
    EXPECT_EQ(local_idx.z, 0);
}

TEST_F(LocalCellInfoTest, GetBgridLocalIdx3D_Offset)
{
    ModuleGint::Vec3i global_idx(startidx_bx + 1, startidx_by + 1, startidx_bz + 1);
    ModuleGint::Vec3i local_idx = localcell_info->get_bgrid_local_idx_3D(global_idx);
    EXPECT_EQ(local_idx.x, 1);
    EXPECT_EQ(local_idx.y, 1);
    EXPECT_EQ(local_idx.z, 1);
}

// Test get_bgrid_local_idx_1D with Vec3i input
TEST_F(LocalCellInfoTest, GetBgridLocalIdx1D_Vec3i)
{
    ModuleGint::Vec3i global_idx(startidx_bx, startidx_by, startidx_bz);
    int local_idx = localcell_info->get_bgrid_local_idx_1D(global_idx);
    EXPECT_EQ(local_idx, 0);
}

// Test get_bgrid_local_idx_1D with int input
TEST_F(LocalCellInfoTest, GetBgridLocalIdx1D_Int)
{
    ModuleGint::Vec3i global_3d(startidx_bx, startidx_by, startidx_bz);
    int global_1d = unitcell_info->bgrid_idx_3Dto1D(global_3d);
    int local_idx = localcell_info->get_bgrid_local_idx_1D(global_1d);
    EXPECT_EQ(local_idx, 0);
}

// Test get_bgrid_global_coord_3D
TEST_F(LocalCellInfoTest, GetBgridGlobalCoord3D_Origin)
{
    ModuleGint::Vec3d coord = localcell_info->get_bgrid_global_coord_3D(0);
    // Biggrid size = 16/4 = 4, starting at (1,1,1)
    EXPECT_DOUBLE_EQ(coord.x, 4.0);
    EXPECT_DOUBLE_EQ(coord.y, 4.0);
    EXPECT_DOUBLE_EQ(coord.z, 4.0);
}

// Test is_bgrid_in_lcell
TEST_F(LocalCellInfoTest, IsBgridInLcell_Inside)
{
    ModuleGint::Vec3i idx(startidx_bx, startidx_by, startidx_bz);
    EXPECT_TRUE(localcell_info->is_bgrid_in_lcell(idx));
    
    idx = ModuleGint::Vec3i(startidx_bx + 1, startidx_by + 1, startidx_bz + 1);
    EXPECT_TRUE(localcell_info->is_bgrid_in_lcell(idx));
}

TEST_F(LocalCellInfoTest, IsBgridInLcell_Outside)
{
    // Before the local cell
    ModuleGint::Vec3i idx(0, 0, 0);
    EXPECT_FALSE(localcell_info->is_bgrid_in_lcell(idx));
    
    // After the local cell
    idx = ModuleGint::Vec3i(startidx_bx + nbx_local, startidx_by, startidx_bz);
    EXPECT_FALSE(localcell_info->is_bgrid_in_lcell(idx));
}

TEST_F(LocalCellInfoTest, IsBgridInLcell_Boundary)
{
    // Just inside the boundary
    ModuleGint::Vec3i idx(startidx_bx + nbx_local - 1, 
                          startidx_by + nby_local - 1, 
                          startidx_bz + nbz_local - 1);
    EXPECT_TRUE(localcell_info->is_bgrid_in_lcell(idx));
}

// Test local meshgrid index conversion
TEST_F(LocalCellInfoTest, MgridIdx_1Dto3D_Origin)
{
    ModuleGint::Vec3i result = localcell_info->mgrid_idx_1Dto3D(0);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(LocalCellInfoTest, MgridIdx_3Dto1D_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    EXPECT_EQ(localcell_info->mgrid_idx_3Dto1D(index_3d), 0);
}

TEST_F(LocalCellInfoTest, MgridIdx_Conversion_Roundtrip)
{
    int nm_local = localcell_info->get_mgrids_num();
    for (int i = 0; i < std::min(nm_local, 27); ++i)  // Test subset
    {
        ModuleGint::Vec3i idx_3d = localcell_info->mgrid_idx_1Dto3D(i);
        EXPECT_EQ(localcell_info->mgrid_idx_3Dto1D(idx_3d), i);
    }
}

// Test get_mgrid_global_idx_3D
TEST_F(LocalCellInfoTest, GetMgridGlobalIdx3D_Origin)
{
    ModuleGint::Vec3i local_idx(0, 0, 0);
    ModuleGint::Vec3i global_idx = localcell_info->get_mgrid_global_idx_3D(local_idx);
    // startidx_mx = startidx_bx * (nm/nb) = 1 * 2 = 2
    EXPECT_EQ(global_idx.x, 2);
    EXPECT_EQ(global_idx.y, 2);
    EXPECT_EQ(global_idx.z, 2);
}

// Test get_mgrid_global_idx_1D
TEST_F(LocalCellInfoTest, GetMgridGlobalIdx1D_Origin)
{
    int global_idx = localcell_info->get_mgrid_global_idx_1D(0);
    ModuleGint::Vec3i expected_3d(2, 2, 2);
    int expected_1d = unitcell_info->mgrid_idx_3Dto1D(expected_3d);
    EXPECT_EQ(global_idx, expected_1d);
}

// Test with local cell at origin
class LocalCellInfoOriginTest : public testing::Test
{
protected:
    void SetUp() override
    {
        unitcell_vec1 = ModuleGint::Vec3d(12.0, 0.0, 0.0);
        unitcell_vec2 = ModuleGint::Vec3d(0.0, 12.0, 0.0);
        unitcell_vec3 = ModuleGint::Vec3d(0.0, 0.0, 12.0);
        
        unitcell_info = std::make_shared<ModuleGint::UnitCellInfo>(
            unitcell_vec1, unitcell_vec2, unitcell_vec3,
            3, 3, 3, 6, 6, 6);

        // Local cell at origin covering all biggrids
        localcell_info = std::make_shared<ModuleGint::LocalCellInfo>(
            0, 0, 0, 3, 3, 3, unitcell_info);
    }

    ModuleGint::Vec3d unitcell_vec1, unitcell_vec2, unitcell_vec3;
    std::shared_ptr<ModuleGint::UnitCellInfo> unitcell_info;
    std::shared_ptr<ModuleGint::LocalCellInfo> localcell_info;
};

TEST_F(LocalCellInfoOriginTest, LocalGlobalConsistency)
{
    // When local cell starts at origin and covers all biggrids,
    // local and global indices should be the same
    for (int i = 0; i < localcell_info->get_bgrids_num(); ++i)
    {
        EXPECT_EQ(localcell_info->get_bgrid_global_idx_1D(i), 
                  unitcell_info->bgrid_idx_3Dto1D(localcell_info->bgrid_idx_1Dto3D(i)));
    }
}

TEST_F(LocalCellInfoOriginTest, AllBgridsInLcell)
{
    for (int x = 0; x < 3; ++x)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int z = 0; z < 3; ++z)
            {
                ModuleGint::Vec3i idx(x, y, z);
                EXPECT_TRUE(localcell_info->is_bgrid_in_lcell(idx));
            }
        }
    }
}
