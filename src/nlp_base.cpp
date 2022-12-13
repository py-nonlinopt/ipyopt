#include "nlp_base.hpp"
#include "IpBlas.hpp"

void arr_copy(const double *src, double *dest, std::size_t n) {
  Ipopt::IpBlasDcopy(n, src, 1, dest, 1);
}

void copy_sparsity(const SparsityIndices &sparsity_indices, Ipopt::Index *iRow,
                   Ipopt::Index *jCol) {
  std::copy(std::get<0>(sparsity_indices).begin(),
            std::get<0>(sparsity_indices).end(), iRow);
  std::copy(std::get<1>(sparsity_indices).begin(),
            std::get<1>(sparsity_indices).end(), jCol);
}

struct OptionVisitor {
  Ipopt::OptionsList &options;
  const char *key;
  OptionVisitor(Ipopt::OptionsList &options, const char *key)
      : options{options}, key{key} {}
  bool operator()(const std::string &val) {
    return options.SetStringValue(key, val);
  }
  bool operator()(const int &val) { return options.SetIntegerValue(key, val); }
  bool operator()(const double &val) {
    return options.SetNumericValue(key, val);
  }
};

NlpData::NlpData(Ipopt::Index n, Ipopt::Index m)
    : _n{n}, _m{m}, _out_x{nullptr}, _out_z_L{nullptr}, _out_z_U{nullptr},
      _out_g{nullptr}, _out_lambda{nullptr}, _out_obj_value{0.}, n{_n}, m{_m},
      out_obj_value{_out_obj_value}, _obj_scaling{0.} {}
void NlpData::set_initial_values(double *x0, double *mult_g, double *mult_x_L,
                                 double *mult_x_U) {
  _out_x = x0;
  _out_lambda = mult_g;
  _out_z_L = mult_x_L;
  _out_z_U = mult_x_U;
}

NlpBundle::NlpBundle() : _app{IpoptApplicationFactory()} {}
NlpBundle::operator bool() { return IsValid(_app); }
int NlpBundle::optimize() { return _app->OptimizeTNLP(_nlp); }

bool NlpBundle::set_option(const char *key, const IpoptOptionValue &value) {
  return std::visit(OptionVisitor(*_app->Options(), key), value);
}
void NlpBundle::without_hess() {
  _app->Options()->SetStringValue("hessian_approximation", "limited-memory");
}

static IpoptOption::Type from_ipopt(const Ipopt::RegisteredOptionType &type) {
  switch (type) {
  case Ipopt::OT_Number:
    return IpoptOption::Number;
  case Ipopt::OT_Integer:
    return IpoptOption::Integer;
  case Ipopt::OT_String:
    return IpoptOption::String;
  default:
    return IpoptOption::Unknown;
  }
}

std::vector<IpoptOption> get_ipopt_options() {
  const auto app = IpoptApplicationFactory();
  const auto options = app->RegOptions()->RegisteredOptionsList();
  auto out = std::vector<IpoptOption>{};
  for (const auto &[key, value] : options) {
    out.push_back(IpoptOption{
        value->Name(), from_ipopt(value->Type()), value->ShortDescription(),
        value->LongDescription(), value->RegisteringCategory()});
  }
  return out;
}
