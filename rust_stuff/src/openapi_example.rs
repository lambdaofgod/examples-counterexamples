extern crate openapi_type;

use openapi_type::OpenapiType;

#[derive(OpenapiType)]
struct FooBar {
    foo: String,
    bar: u64,
}

pub fn main() {
    println!("{:#?}", FooBar::schema());
}
