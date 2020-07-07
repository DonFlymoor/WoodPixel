/** \file simple_cl_error.h
*	\author Fabian Friederichs
*
*	\brief Customized exception class and error handling macros.
*/

#ifndef _ID_GEN_H_
#define _ID_GEN_H_
#include <cstddef>
#include <atomic>


/**
\brief A simple ID generator based on an atomic counter.

0 is always an invalid ID.
*/
class IdGen
{
private:
	std::atomic<std::size_t> m_idct;

public:
	IdGen() noexcept :
		m_idct(1)
	{
	}

	/**
	\brief Returns a new ID that is unique as long as the generator is not reset.

	This function is thread safe and lock-free.
	*/
	std::size_t createID() noexcept
	{
		return m_idct.fetch_add(1, std::memory_order_relaxed);
	}

	/**
	\brief Resets the atomic counter.
	\attention	Don't call this function if you don't own the generator. This would break uniqueness between IDs if any IDs that were generated before the reset are still used.
	*/
	void reset() noexcept
	{
		m_idct.store(1, std::memory_order_relaxed);
	}
};

#endif