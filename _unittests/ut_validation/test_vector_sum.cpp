#include "onnx-extended/validation/vector_function.h"

#define ASSERT_THROW( condition ) \
{                                                                   \
  if( !( condition ) ) {                                            \
    throw std::runtime_error(   std::string( __FILE__ )             \
                              + std::string( ":" )                  \
                              + std::to_string( __LINE__ )          \
                              + std::string( " in " )               \
                              + std::string( __PRETTY_FUNCTION__ )  \
    );                                                              \
  }                                                                 \
}

void testAssertTrue() {
  ASSERT_THROW( true );
}

void test_vector_sum() {
    std::vector<float> m(10);
    for(size_t i =0;i<m.size();++i) {
        m[i] = static_cast<float>(i);
    }
    float s1 = vector_sum(2, 5, &m[0], true);
    float s2 = vector_sum(2, 5, &m[0], false);
    ASSERT_THROW(s1 == s2);
}


int main(int, char**) {
  testAssertTrue();
  test_vector_sum();
}
