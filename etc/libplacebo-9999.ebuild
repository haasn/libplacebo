# Copyright 1999-2017 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2

EAPI=6

if [[ "${PV}" == "9999" ]]; then
	EGIT_REPO_URI="https://github.com/haasn/libplacebo"
	inherit git-r3
else
	KEYWORDS="~amd64"
	SRC_URI="https://github.com/haasn/libplacebo/archive/v${PV}.tar.gz -> ${P}.tar.gz"
fi

inherit meson multilib-minimal

DESCRIPTION="Reusable library for GPU-accelerated image processing primitives"
HOMEPAGE="https://github.com/haasn/libplacebo"

LICENSE="LGPLv2.1+"
SLOT="0"
IUSE="shaderc vulkan"

RDEPEND="shaderc? ( dev-util/shaderc[${MULTILIB_USEDEP}] )
	vulkan? ( media-libs/vulkan-loader[${MULTILIB_USEDEP}] )"
DEPEND="${RDEPEND}"

DOCS="README.md"

multilib_src_configure() {
	local emesonargs=(
		-D shaderc=$(usex shaderc true false)
		-D vulkan=$(usex vulkan true false)
	)
	meson_src_configure
}

multilib_src_compile() {
	eninja
}

multilib_src_install() {
	DESTDIR="${D}" eninja install
}
