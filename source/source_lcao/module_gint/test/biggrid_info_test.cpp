#include "../biggrid_info.h"
#include "gtest/gtest.h"
#include <cmath>

/************************************************
 *  unit test of BigGridInfo class
 ***********************************************/

/**
 * Tested functions of class BigGridInfo:
 *  - constructor
 *  - get_cartesian_coord: convert 3D index to Cartesian coordinate
 *  - get_direct_coord: convert Cartesian coordinate to direct coordinate
 *  - max_ext_bgrid_num: calculate maximum extension number for a given radius
 *  - get_nmx, get_nmy, get_nmz, get_mgrids_num: getter functions for meshgrid dimensions
 *  - get_mgrids_coord, get_mgrid_coord: getter functions for meshgrid coordinates
 *  - mgrid_idx_1Dto3D, mgrid_idx_3Dto1D: meshgrid index conversion
 */

class BigGridInfoTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple cubic biggrid with vectors (4,0,0), (0,4,0), (0,0,4)
        // with 2x2x2 meshgrids
        biggrid_vec1 = ModuleGint::Vec3d(4.0, 0.0, 0.0);
        biggrid_vec2 = ModuleGint::Vec3d(0.0, 4.0, 0.0);
        biggrid_vec3 = ModuleGint::Vec3d(0.0, 0.0, 4.0);
        nmx = 2;
        nmy = 2;
        nmz = 2;
        
        biggrid_info = std::make_shared<ModuleGint::BigGridInfo>(
            biggrid_vec1, biggrid_vec2, biggrid_vec3, nmx, nmy, nmz);
    }

    ModuleGint::Vec3d biggrid_vec1, biggrid_vec2, biggrid_vec3;
    int nmx, nmy, nmz;
    std::shared_ptr<ModuleGint::BigGridInfo> biggrid_info;
};

// Test constructor and getter functions
TEST_F(BigGridInfoTest, Constructor)
{
    EXPECT_EQ(biggrid_info->get_nmx(), nmx);
    EXPECT_EQ(biggrid_info->get_nmy(), nmy);
    EXPECT_EQ(biggrid_info->get_nmz(), nmz);
    EXPECT_EQ(biggrid_info->get_mgrids_num(), nmx * nmy * nmz);
}

// Test get_cartesian_coord with Vec3d
TEST_F(BigGridInfoTest, GetCartesianCoord_Vec3d_Origin)
{
    ModuleGint::Vec3d index_3d(0.0, 0.0, 0.0);
    ModuleGint::Vec3d coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

TEST_F(BigGridInfoTest, GetCartesianCoord_Vec3d_Unit)
{
    ModuleGint::Vec3d index_3d(1.0, 0.0, 0.0);
    ModuleGint::Vec3d coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 4.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
    
    index_3d = ModuleGint::Vec3d(0.0, 1.0, 0.0);
    coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 4.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
    
    index_3d = ModuleGint::Vec3d(0.0, 0.0, 1.0);
    coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 4.0);
}

// Test get_cartesian_coord with Vec3i
TEST_F(BigGridInfoTest, GetCartesianCoord_Vec3i_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    ModuleGint::Vec3d coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

TEST_F(BigGridInfoTest, GetCartesianCoord_Vec3i_Unit)
{
    ModuleGint::Vec3i index_3d(1, 2, 3);
    ModuleGint::Vec3d coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_DOUBLE_EQ(coord.x, 4.0);
    EXPECT_DOUBLE_EQ(coord.y, 8.0);
    EXPECT_DOUBLE_EQ(coord.z, 12.0);
}

// Test get_direct_coord
TEST_F(BigGridInfoTest, GetDirectCoord_Origin)
{
    ModuleGint::Vec3d cart_coord(0.0, 0.0, 0.0);
    ModuleGint::Vec3d direct_coord = biggrid_info->get_direct_coord(cart_coord);
    EXPECT_NEAR(direct_coord.x, 0.0, 1e-10);
    EXPECT_NEAR(direct_coord.y, 0.0, 1e-10);
    EXPECT_NEAR(direct_coord.z, 0.0, 1e-10);
}

TEST_F(BigGridInfoTest, GetDirectCoord_Unit)
{
    ModuleGint::Vec3d cart_coord(4.0, 0.0, 0.0);
    ModuleGint::Vec3d direct_coord = biggrid_info->get_direct_coord(cart_coord);
    EXPECT_NEAR(direct_coord.x, 1.0, 1e-10);
    EXPECT_NEAR(direct_coord.y, 0.0, 1e-10);
    EXPECT_NEAR(direct_coord.z, 0.0, 1e-10);
    
    cart_coord = ModuleGint::Vec3d(0.0, 8.0, 0.0);
    direct_coord = biggrid_info->get_direct_coord(cart_coord);
    EXPECT_NEAR(direct_coord.x, 0.0, 1e-10);
    EXPECT_NEAR(direct_coord.y, 2.0, 1e-10);
    EXPECT_NEAR(direct_coord.z, 0.0, 1e-10);
}

// Test roundtrip conversion
TEST_F(BigGridInfoTest, CoordConversion_Roundtrip)
{
    ModuleGint::Vec3d index_3d(1.5, 2.5, 3.5);
    ModuleGint::Vec3d cart_coord = biggrid_info->get_cartesian_coord(index_3d);
    ModuleGint::Vec3d result = biggrid_info->get_direct_coord(cart_coord);
    EXPECT_NEAR(result.x, index_3d.x, 1e-10);
    EXPECT_NEAR(result.y, index_3d.y, 1e-10);
    EXPECT_NEAR(result.z, index_3d.z, 1e-10);
}

// Test max_ext_bgrid_num
TEST_F(BigGridInfoTest, MaxExtBgridNum_SmallRadius)
{
    double r = 2.0;  // Half of the biggrid size
    ModuleGint::Vec3i ext_num = biggrid_info->max_ext_bgrid_num(r);
    // For orthogonal vectors, ext = floor(r / |vec|) + 1 = floor(2/4) + 1 = 1
    EXPECT_EQ(ext_num.x, 1);
    EXPECT_EQ(ext_num.y, 1);
    EXPECT_EQ(ext_num.z, 1);
}

TEST_F(BigGridInfoTest, MaxExtBgridNum_LargeRadius)
{
    double r = 10.0;
    ModuleGint::Vec3i ext_num = biggrid_info->max_ext_bgrid_num(r);
    // For orthogonal vectors with |vec| = 4, ext = floor(10/4) + 1 = 3
    EXPECT_EQ(ext_num.x, 3);
    EXPECT_EQ(ext_num.y, 3);
    EXPECT_EQ(ext_num.z, 3);
}

// Test meshgrid index conversion
TEST_F(BigGridInfoTest, MgridIdx_1Dto3D_Origin)
{
    ModuleGint::Vec3i result = biggrid_info->mgrid_idx_1Dto3D(0);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(BigGridInfoTest, MgridIdx_3Dto1D_Origin)
{
    ModuleGint::Vec3i index_3d(0, 0, 0);
    EXPECT_EQ(biggrid_info->mgrid_idx_3Dto1D(index_3d), 0);
}

TEST_F(BigGridInfoTest, MgridIdx_Conversion_Roundtrip)
{
    for (int i = 0; i < nmx * nmy * nmz; ++i)
    {
        ModuleGint::Vec3i idx_3d = biggrid_info->mgrid_idx_1Dto3D(i);
        EXPECT_EQ(biggrid_info->mgrid_idx_3Dto1D(idx_3d), i);
    }
}

// Test meshgrid coordinates
TEST_F(BigGridInfoTest, GetMgridsCoord)
{
    const std::vector<ModuleGint::Vec3d>& coords = biggrid_info->get_mgrids_coord();
    EXPECT_EQ(coords.size(), static_cast<size_t>(nmx * nmy * nmz));
}

TEST_F(BigGridInfoTest, GetMgridCoord_Origin)
{
    const ModuleGint::Vec3d& coord = biggrid_info->get_mgrid_coord(0);
    EXPECT_DOUBLE_EQ(coord.x, 0.0);
    EXPECT_DOUBLE_EQ(coord.y, 0.0);
    EXPECT_DOUBLE_EQ(coord.z, 0.0);
}

// Test with non-orthogonal biggrid vectors
class BigGridInfoNonOrthTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // Create a non-orthogonal biggrid
        biggrid_vec1 = ModuleGint::Vec3d(4.0, 0.0, 0.0);
        biggrid_vec2 = ModuleGint::Vec3d(2.0, 3.464, 0.0);  // 60 degree angle
        biggrid_vec3 = ModuleGint::Vec3d(0.0, 0.0, 4.0);
        nmx = 2;
        nmy = 2;
        nmz = 2;
        
        biggrid_info = std::make_shared<ModuleGint::BigGridInfo>(
            biggrid_vec1, biggrid_vec2, biggrid_vec3, nmx, nmy, nmz);
    }

    ModuleGint::Vec3d biggrid_vec1, biggrid_vec2, biggrid_vec3;
    int nmx, nmy, nmz;
    std::shared_ptr<ModuleGint::BigGridInfo> biggrid_info;
};

TEST_F(BigGridInfoNonOrthTest, GetCartesianCoord_NonOrth)
{
    ModuleGint::Vec3i index_3d(1, 1, 0);
    ModuleGint::Vec3d coord = biggrid_info->get_cartesian_coord(index_3d);
    EXPECT_NEAR(coord.x, 6.0, 1e-10);
    EXPECT_NEAR(coord.y, 3.464, 1e-3);
    EXPECT_NEAR(coord.z, 0.0, 1e-10);
}
