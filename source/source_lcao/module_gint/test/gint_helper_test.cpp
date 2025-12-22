#include "../gint_helper.h"
#include "gtest/gtest.h"

/************************************************
 *  unit test of gint_helper functions
 ***********************************************/

/**
 * Tested functions:
 *  - index3Dto1D: convert 3D index to 1D index
 *  - index1Dto3D: convert 1D index to 3D index
 *  - pow_int: fast integer power for exponents 0-5
 *  - floor_div: floor division for integers
 *  - ceil_div: ceiling division for integers
 */

class GintHelperTest : public testing::Test
{
protected:
    void SetUp() override
    {
        // dimensions for index conversion tests
        dim_x = 3;
        dim_y = 4;
        dim_z = 5;
    }

    int dim_x, dim_y, dim_z;
};

// Test index3Dto1D function
TEST_F(GintHelperTest, Index3Dto1D_Origin)
{
    EXPECT_EQ(ModuleGint::index3Dto1D(0, 0, 0, dim_x, dim_y, dim_z), 0);
}

TEST_F(GintHelperTest, Index3Dto1D_ZOnly)
{
    EXPECT_EQ(ModuleGint::index3Dto1D(0, 0, 1, dim_x, dim_y, dim_z), 1);
    EXPECT_EQ(ModuleGint::index3Dto1D(0, 0, 4, dim_x, dim_y, dim_z), 4);
}

TEST_F(GintHelperTest, Index3Dto1D_YOnly)
{
    EXPECT_EQ(ModuleGint::index3Dto1D(0, 1, 0, dim_x, dim_y, dim_z), dim_z);
    EXPECT_EQ(ModuleGint::index3Dto1D(0, 3, 0, dim_x, dim_y, dim_z), 3 * dim_z);
}

TEST_F(GintHelperTest, Index3Dto1D_XOnly)
{
    EXPECT_EQ(ModuleGint::index3Dto1D(1, 0, 0, dim_x, dim_y, dim_z), dim_y * dim_z);
    EXPECT_EQ(ModuleGint::index3Dto1D(2, 0, 0, dim_x, dim_y, dim_z), 2 * dim_y * dim_z);
}

TEST_F(GintHelperTest, Index3Dto1D_Combined)
{
    int id_x = 1, id_y = 2, id_z = 3;
    int expected = id_z + id_y * dim_z + id_x * dim_y * dim_z;
    EXPECT_EQ(ModuleGint::index3Dto1D(id_x, id_y, id_z, dim_x, dim_y, dim_z), expected);
}

TEST_F(GintHelperTest, Index3Dto1D_MaxIndex)
{
    int expected = dim_x * dim_y * dim_z - 1;
    EXPECT_EQ(ModuleGint::index3Dto1D(dim_x - 1, dim_y - 1, dim_z - 1, dim_x, dim_y, dim_z), expected);
}

// Test index1Dto3D function
TEST_F(GintHelperTest, Index1Dto3D_Origin)
{
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(0, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(GintHelperTest, Index1Dto3D_ZOnly)
{
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(3, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 3);
}

TEST_F(GintHelperTest, Index1Dto3D_YOnly)
{
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(dim_z, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, 0);
    EXPECT_EQ(result.y, 1);
    EXPECT_EQ(result.z, 0);
}

TEST_F(GintHelperTest, Index1Dto3D_XOnly)
{
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(dim_y * dim_z, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, 1);
    EXPECT_EQ(result.y, 0);
    EXPECT_EQ(result.z, 0);
}

TEST_F(GintHelperTest, Index1Dto3D_Combined)
{
    int id_x = 1, id_y = 2, id_z = 3;
    int index_1d = id_z + id_y * dim_z + id_x * dim_y * dim_z;
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(index_1d, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, id_x);
    EXPECT_EQ(result.y, id_y);
    EXPECT_EQ(result.z, id_z);
}

TEST_F(GintHelperTest, Index1Dto3D_MaxIndex)
{
    int max_index = dim_x * dim_y * dim_z - 1;
    ModuleGint::Vec3i result = ModuleGint::index1Dto3D(max_index, dim_x, dim_y, dim_z);
    EXPECT_EQ(result.x, dim_x - 1);
    EXPECT_EQ(result.y, dim_y - 1);
    EXPECT_EQ(result.z, dim_z - 1);
}

// Test roundtrip conversion
TEST_F(GintHelperTest, IndexConversion_Roundtrip)
{
    for (int x = 0; x < dim_x; ++x)
    {
        for (int y = 0; y < dim_y; ++y)
        {
            for (int z = 0; z < dim_z; ++z)
            {
                int index_1d = ModuleGint::index3Dto1D(x, y, z, dim_x, dim_y, dim_z);
                ModuleGint::Vec3i result = ModuleGint::index1Dto3D(index_1d, dim_x, dim_y, dim_z);
                EXPECT_EQ(result.x, x);
                EXPECT_EQ(result.y, y);
                EXPECT_EQ(result.z, z);
            }
        }
    }
}

// Test pow_int function
TEST(GintHelper, PowInt_Exp0)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 0), 1.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.5, 0), 1.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-1.5, 0), 1.0);
}

TEST(GintHelper, PowInt_Exp1)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 1), 2.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.5, 1), 3.5);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-1.5, 1), -1.5);
}

TEST(GintHelper, PowInt_Exp2)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 2), 4.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.0, 2), 9.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-2.0, 2), 4.0);
}

TEST(GintHelper, PowInt_Exp3)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 3), 8.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.0, 3), 27.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-2.0, 3), -8.0);
}

TEST(GintHelper, PowInt_Exp4)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 4), 16.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.0, 4), 81.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-2.0, 4), 16.0);
}

TEST(GintHelper, PowInt_Exp5)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 5), 32.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(3.0, 5), 243.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(-2.0, 5), -32.0);
}

TEST(GintHelper, PowInt_ExpLarger)
{
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 6), 64.0);
    EXPECT_DOUBLE_EQ(ModuleGint::pow_int(2.0, 10), 1024.0);
}

// Test floor_div function
TEST(GintHelper, FloorDiv_BothPositive)
{
    EXPECT_EQ(ModuleGint::floor_div(7, 3), 2);
    EXPECT_EQ(ModuleGint::floor_div(9, 3), 3);
    EXPECT_EQ(ModuleGint::floor_div(10, 3), 3);
}

TEST(GintHelper, FloorDiv_NegativeDividend)
{
    EXPECT_EQ(ModuleGint::floor_div(-7, 3), -3);
    EXPECT_EQ(ModuleGint::floor_div(-9, 3), -3);
    EXPECT_EQ(ModuleGint::floor_div(-10, 3), -4);
}

TEST(GintHelper, FloorDiv_NegativeDivisor)
{
    EXPECT_EQ(ModuleGint::floor_div(7, -3), -3);
    EXPECT_EQ(ModuleGint::floor_div(9, -3), -3);
    EXPECT_EQ(ModuleGint::floor_div(10, -3), -4);
}

TEST(GintHelper, FloorDiv_BothNegative)
{
    EXPECT_EQ(ModuleGint::floor_div(-7, -3), 2);
    EXPECT_EQ(ModuleGint::floor_div(-9, -3), 3);
    EXPECT_EQ(ModuleGint::floor_div(-10, -3), 3);
}

TEST(GintHelper, FloorDiv_ZeroDividend)
{
    EXPECT_EQ(ModuleGint::floor_div(0, 3), 0);
    EXPECT_EQ(ModuleGint::floor_div(0, -3), 0);
}

// Test ceil_div function
TEST(GintHelper, CeilDiv_BothPositive)
{
    EXPECT_EQ(ModuleGint::ceil_div(7, 3), 3);
    EXPECT_EQ(ModuleGint::ceil_div(9, 3), 3);
    EXPECT_EQ(ModuleGint::ceil_div(10, 3), 4);
}

TEST(GintHelper, CeilDiv_NegativeDividend)
{
    EXPECT_EQ(ModuleGint::ceil_div(-7, 3), -2);
    EXPECT_EQ(ModuleGint::ceil_div(-9, 3), -3);
    EXPECT_EQ(ModuleGint::ceil_div(-10, 3), -3);
}

TEST(GintHelper, CeilDiv_NegativeDivisor)
{
    EXPECT_EQ(ModuleGint::ceil_div(7, -3), -2);
    EXPECT_EQ(ModuleGint::ceil_div(9, -3), -3);
    EXPECT_EQ(ModuleGint::ceil_div(10, -3), -3);
}

TEST(GintHelper, CeilDiv_BothNegative)
{
    EXPECT_EQ(ModuleGint::ceil_div(-7, -3), 3);
    EXPECT_EQ(ModuleGint::ceil_div(-9, -3), 3);
    EXPECT_EQ(ModuleGint::ceil_div(-10, -3), 4);
}

TEST(GintHelper, CeilDiv_ZeroDividend)
{
    EXPECT_EQ(ModuleGint::ceil_div(0, 3), 0);
    EXPECT_EQ(ModuleGint::ceil_div(0, -3), 0);
}
